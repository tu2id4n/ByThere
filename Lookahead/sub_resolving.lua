--- Implements depth-limited re-solving at a node of the game tree.
-- Internally uses @{cfrd_gadget|CFRDGadget} TODO SOLVER
-- @classmod SubResolving
require 'Lookahead.sub_lookahead'
require 'Lookahead.cfrd_gadget'
require 'Tree.tree_builder'
require 'Tree.tree_visualiser'
local arguments = require 'Settings.arguments'
local constants = require 'Settings.constants'
local tools = require 'tools'
local card_tools = require 'Game.card_tools'

local SubResolving = torch.class('SubResolving')

--- Constructor
function SubResolving:__init(terminal_equity)
  self.tree_builder = PokerTreeBuilder()
  self.terminal_equity = terminal_equity
end

--- Builds a depth-limited public tree rooted at a given game node.
-- @param node the root of the tree
-- @local
function SubResolving:_create_lookahead_tree(node)
  local build_tree_params = {}
  build_tree_params.root_node = node
  -- guo change to false to remove network
  -- add argument 
  build_tree_params.limit_to_street = arguments.limit_to_street
  self.lookahead_tree = self.tree_builder:build_tree(build_tree_params)
end

--- Re-solves a depth-limited lookahead using input ranges.
--
-- Uses the input range for the opponent instead of a gadget range, so only
-- appropriate for re-solving the root node of the game tree (where ranges
-- are fixed).
--
-- @param node the public node at which to re-solve
-- @param player_range a range vector for the re-solving player
-- @param opponent_range a range vector for the opponent
function SubResolving:resolve_first_node(node, player_range, opponent_range)
  -- print('--------------------------------------------------')
  self.player_range = player_range
  self.opponent_range = opponent_range
  self.opponent_cfvs = nil
  self:_create_lookahead_tree(node)
  
  if player_range:dim() == 1 then
    player_range = player_range:view(1, player_range:size(1))
    opponent_range = opponent_range:view(1, opponent_range:size(1))
  end
  self.lookahead = SubLookahead(self.terminal_equity, player_range:size(1))

  -- local timer = torch.Timer()
  -- timer:reset()
  self.lookahead:build_lookahead(self.lookahead_tree)

  -- print('sub build time: ' .. timer:time().real)
  -- timer:reset()

  self.lookahead:resolve_first_node(player_range, opponent_range)
  -- print('sub resolve time: ' .. timer:time().real)

  self.resolve_results = self.lookahead:get_results()

  -- print('--------------------------------------------------')
  return self.resolve_results
end

--- Re-solves a depth-limited lookahead using an input range for the player and
-- the @{cfrd_gadget|CFRDGadget} to generate ranges for the opponent.
--
-- @param node the public node at which to re-solve
-- @param player_range a range vector for the re-solving player
-- @param opponent_cfvs a vector of cfvs achieved by the opponent
-- before re-solving
function SubResolving:resolve(node, player_range, opponent_cfvs)
  assert(card_tools:is_valid_range(player_range, node.board))

  local timer = torch.Timer()
  timer:reset()
  self.player_range = player_range
  self.opponent_cfvs = opponent_cfvs
  self:_create_lookahead_tree(node)

  if player_range:dim() == 1 then
    player_range = player_range:view(1, player_range:size(1))
  end
  self.lookahead = SubLookahead(self.terminal_equity, player_range:size(1))

  self.lookahead:build_lookahead(self.lookahead_tree)

  -- print('sub build time: ' .. timer:time().real)
  timer:reset()
  self.lookahead:resolve(player_range, opponent_cfvs)

  -- print('sub resolve time: ' .. timer:time().real)
  self.resolve_results = self.lookahead:get_results()
  return self.resolve_results
end

--- Gives the index of the given action at the node being re-solved.
--
-- The node must first be re-solved with @{resolve} or @{resolve_first_node}.
-- @param action a legal action at the node
-- @return the index of the action
-- @local
function SubResolving:_action_to_action_id(action)
  local actions = self:get_possible_actions(action)
  local action_id = -1
  for i=1,actions:size(1) do
    if action == actions[i] then
      action_id = i
    end
  end

  assert(action_id ~= -1)

  return action_id
end

--- Gives a list of possible actions at the node being re-solved.
--
-- The node must first be re-solved with @{resolve} or @{resolve_first_node}.
-- @return a list of legal actions
function SubResolving:get_possible_actions()
  return self.lookahead_tree.actions
end

--- Gives the average counterfactual values that the re-solve player received
-- at the node during re-solving.
--
-- The node must first be re-solved with @{resolve_first_node}.
--
-- @return a vector of cfvs
function SubResolving:get_root_cfv()
  return self.resolve_results.root_cfvs
end

--- Gives the average counterfactual values that each player received
-- at the node during re-solving.
--
-- Usefull for data generation for neural net training
--
-- The node must first be re-solved with @{resolve_first_node}.
--
-- @return a 2xK tensor of cfvs, where K is the range size
function SubResolving:get_root_cfv_both_players()
  return self.resolve_results.root_cfvs_both_players
end

--- Gives the average counterfactual values that the opponent received
-- during re-solving after the re-solve player took a given action.
--
-- Used during continual re-solving to track opponent cfvs. The node must
-- first be re-solved with @{resolve} or @{resolve_first_node}.
--
-- @param action the action taken by the re-solve player at the node being
-- re-solved
-- @return a vector of cfvs
function SubResolving:get_action_cfv(action)
  local action_id = self:_action_to_action_id(action)
  return self.resolve_results.children_cfvs[action_id]
end

--- Gives the average counterfactual values that the opponent received
-- during re-solving after a chance event (the betting round changes and
-- more cards are dealt).
--
-- Used during continual re-solving to track opponent cfvs. The node must
-- first be re-solved with @{resolve} or @{resolve_first_node}.
--
-- @param action the action taken by the re-solve player at the node being
-- re-solved
-- @param board a vector of board cards which were updated by the chance event
-- @return a vector of cfvs
function SubResolving:get_chance_action_cfv(action, board)
  -- resolve to get next_board chance actions if flop
  if board:dim() == 1 and board:size(1) == 3 then
    self.lookahead:reset()

    local board_idx = card_tools:get_flop_board_index(board)
    self.lookahead.next_board_idx = board_idx

    if self.opponent_cfvs ~= nil then
      self.lookahead:resolve(self.player_range, self.opponent_cfvs)
    else
      self.lookahead:resolve_first_node(self.player_range, self.opponent_range)
    end

    self.lookahead.next_board_idx = nil
  end
  return self.lookahead:get_chance_action_cfv(action, board)
end

--- Gives the probability that the re-solved strategy takes a given action.
--
-- The node must first be re-solved with @{resolve} or @{resolve_first_node}.
--
-- @param action a legal action at the re-solve node
-- @return a vector giving the probability of taking the action with each
-- private hand
function SubResolving:get_action_strategy(action)
  local action_id = self:_action_to_action_id(action)
  return self.resolve_results.strategy[action_id][1]
end

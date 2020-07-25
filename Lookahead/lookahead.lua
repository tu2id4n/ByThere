--- A depth-limited lookahead of the game tree used for re-solving.
-- @classmod lookahead

require 'Lookahead.lookahead_builder'
require 'TerminalEquity.terminal_equity'
require 'Lookahead.cfrd_gadget'
local arguments = require 'Settings.arguments'
local constants = require 'Settings.constants'
local game_settings = require 'Settings.game_settings'
local tools = require 'tools'
local card_tools = require 'Game.card_tools'
local card_to_string = require 'Game.card_to_string_conversion'



local Lookahead = torch.class('Lookahead')
local timings = {}
--- Constructor
function Lookahead:__init(terminal_equity, batch_size)
  self.builder = LookaheadBuilder(self)
  self.terminal_equity = terminal_equity
  self.batch_size = batch_size
  --v2
  -- print('init')
  -- print(terminal_equity.board)
  local next_boards = card_tools:get_last_round_boards(terminal_equity.board)
  self.next_boards = next_boards

  self.te = TerminalEquity()
  -- print('self.next_boards', self.next_boards)
end

-- v3 深拷贝，优化启动速度
function Lookahead:deep_copy_te()
  local lookup_table = {}
    local function _copy(object)
        if type(object) ~= "table" then
            return object
        elseif lookup_table[object] then
            return lookup_table[object]
        end
        local new_table = {}
        lookup_table[object] = new_table
        for key, value in pairs(object) do
            new_table[_copy(key)] = _copy(value)
        end
        return setmetatable(new_table, getmetatable(object))
    end
    return _copy(self.te)
end
--- Constructs the lookahead from a game's public tree.
--
-- Must be called to initialize the lookahead.
-- @param tree a public tree
function Lookahead:build_lookahead(tree)
  self.builder:build_from_tree(tree)
end

function Lookahead:reset()
  self.builder:reset()
end
--- Re-solves the lookahead using input ranges.
--
-- Uses the input range for the opponent instead of a gadget range, so only
-- appropriate for re-solving the root node of the game tree (where ranges
-- are fixed).
--
-- @{build_lookahead} must be called first.
--
-- @param player_range a range vector for the re-solving player
-- @param opponent_range a range vector for the opponent
function Lookahead:resolve_first_node(player_range, opponent_range)
  -- print("--pot--")
  -- print(self.next_round_pot_sizes)
  -- print(self.pot_size)
  -- print(self.indices)
  -- local test = self.next_round_pot_sizes:view(-1, 1):clone()
  -- test = test:expand(test:size(1), arguments.gen_batch_size):clone()
  -- test = test:view(-1, 1)
  -- print(test)

  -- print("-------")
  -- print("term_call: ", self.term_call_indices)
  -- print("indices: ", self.indices)
  self.ranges_data[1][{{}, {}, {}, {}, 1, {}}]:copy(player_range)
  self.ranges_data[1][{{}, {}, {}, {}, 2, {}}]:copy(opponent_range)
  local timer = torch.Timer()
  timer:reset()
  self:_compute()
  print('total compute time in lookahead: ' .. timer:time().real)
end

--- Re-solves the lookahead using an input range for the player and
-- the @{cfrd_gadget|CFRDGadget} to generate ranges for the opponent.
--
-- @{build_lookahead} must be called first.
--
-- @param player_range a range vector for the re-solving player
-- @param opponent_cfvs a vector of cfvs achieved by the opponent
-- before re-solving
function Lookahead:resolve(player_range, opponent_cfvs)
  assert(player_range)
  assert(opponent_cfvs)

  self.reconstruction_gadget = CFRDGadget(self.tree.board, player_range, opponent_cfvs)

  self.ranges_data[1][{{}, {}, {}, {}, 1, {}}]:copy(player_range)
  self.reconstruction_opponent_cfvs = opponent_cfvs
  self:_compute()
end

--- Re-solves the lookahead.
-- @local
function Lookahead:_compute()
  --1.0 main loop

  for i=1,8 do
    timings[i] = 0
  end
  for iter=1,arguments.cfr_iters do

    local timer = torch.Timer()
    timer:reset()
    self:_set_opponent_starting_range(iter)
    timings[1] = timings[1] + timer:time().real
    timer:reset()
    self:_compute_current_strategies()
    timings[2] = timings[2] + timer:time().real
    timer:reset()
    self:_compute_ranges()
    timings[3] = timings[3] + timer:time().real
    timer:reset()
    self:_compute_update_average_strategies(iter)
    timings[4] = timings[4] + timer:time().real
    timer:reset()
    self:_compute_terminal_equities(iter)
    timings[5] = timings[5] + timer:time().real
    timer:reset()
    self:_compute_cfvs()
    timings[6] = timings[6] + timer:time().real
    timer:reset()
    self:_compute_regrets()
    timings[7] = timings[7] + timer:time().real
    timer:reset()
    self:_compute_cumulate_average_cfvs(iter)
    timings[8] = timings[8] + timer:time().real
    timer:reset()
  end

  for i=1,8 do
    print(' ' .. i .. ': ' .. timings[i])
  end
  local timer = torch.Timer()
  timer:reset()
  --2.0 at the end normalize average strategy
  self:_compute_normalize_average_strategies()
  --2.1 normalize root's CFVs
  self:_compute_normalize_average_cfvs()
  print('other time in iter compute: ' .. timer:time().real)
end

--- Uses regret matching to generate the players' current strategies.
-- @local
function Lookahead:_compute_current_strategies()
  for d=2,self.depth do
    self.positive_regrets_data[d]:copy(self.regrets_data[d])
    self.positive_regrets_data[d]:clamp(self.regret_epsilon, tools:max_number())

    --1.0 set regret of empty actions to 0
    self.positive_regrets_data[d]:cmul(self.empty_action_mask[d])

    --1.1  regret matching
    --note that the regrets as well as the CFVs have switched player indexing
    torch.sum(self.regrets_sum[d], self.positive_regrets_data[d], 1)
    local player_current_strategy = self.current_strategy_data[d]
    local player_regrets = self.positive_regrets_data[d]
    local player_regrets_sum = self.regrets_sum[d]

    player_current_strategy:cdiv(player_regrets, player_regrets_sum:expandAs(player_regrets))
  end
end

--- Using the players' current strategies, computes their probabilities of
-- reaching each state of the lookahead.
-- @local
function Lookahead:_compute_ranges()

  for d=1,self.depth-1 do
    local current_level_ranges = self.ranges_data[d]
    local next_level_ranges = self.ranges_data[d+1]

    local prev_layer_terminal_actions_count = self.terminal_actions_count[d-1]
    local prev_layer_actions_count = self.actions_count[d-1]
    local prev_layer_bets_count = self.bets_count[d-1]
    local gp_layer_nonallin_bets_count = self.nonallinbets_count[d-2]
    local gp_layer_terminal_actions_count = self.terminal_actions_count[d-2]


    --copy the ranges of inner nodes and transpose
    self.inner_nodes[d]:copy(current_level_ranges[{{prev_layer_terminal_actions_count+1, -1}, {1, gp_layer_nonallin_bets_count}, {}, {}, {}, {}}]:transpose(2,3))

    local super_view = self.inner_nodes[d]
    super_view = super_view:view(1, prev_layer_bets_count, -1, self.batch_size, constants.players_count, game_settings.hand_count)

    super_view = super_view:expandAs(next_level_ranges)
    local next_level_strategies = self.current_strategy_data[d+1]

    next_level_ranges:copy(super_view)

    --multiply the ranges of the acting player by his strategy
    next_level_ranges[{{}, {}, {}, {}, self.acting_player[d], {}}]:cmul(next_level_strategies)
  end
end

--- Updates the players' average strategies with their current strategies.
-- @param iter the current iteration number of re-solving
-- @local
function Lookahead:_compute_update_average_strategies(iter)
  if iter > arguments.cfr_skip_iters then
    --no need to go through layers since we care for the average strategy only in the first node anyway
    --note that if you wanted to average strategy on lower layers, you would need to weight the current strategy by the current reach probability
    self.average_strategies_data[2]:add(self.current_strategy_data[2])
  end
end

--- Using the players' reach probabilities, computes their counterfactual
-- values at each lookahead state which is a terminal state of the game.
-- @local
function Lookahead:_compute_terminal_equities_terminal_equity()

  -- copy in range data
  for d=2,self.depth do
    if d > 2 or self.first_call_terminal then
      if self.tree.street ~= constants.streets_count then
        self.ranges_data_call[{self.term_call_indices[d]}]:copy(self.ranges_data[d][2][-1])
      else
        self.ranges_data_call[{self.term_call_indices[d]}]:copy(self.ranges_data[d][2])
      end
    end
    self.ranges_data_fold[{self.term_fold_indices[d]}]:copy(self.ranges_data[d][1])
  end

  self.terminal_equity:call_value(self.ranges_data_call:view(-1, game_settings.hand_count), self.cfvs_data_call:view(-1, game_settings.hand_count))
  self.terminal_equity:fold_value(self.ranges_data_fold:view(-1, game_settings.hand_count), self.cfvs_data_fold:view(-1, game_settings.hand_count))

  for d=2,self.depth do
    if self.tree.street ~= constants.streets_count then
      if game_settings.nl and (d>2 or self.first_call_terminal) then
        self.cfvs_data[d][2][-1]:copy(self.cfvs_data_call[{self.term_call_indices[d]}])
        -- print("Termial: cfvs_data: [" .. d .. "][2][-1],  = ", self.cfvs_data_call[{self.term_call_indices[d]}]:size())
      end
    else
      if d>2 or self.first_call_terminal then
        self.cfvs_data[d][2]:copy(self.cfvs_data_call[{self.term_call_indices[d]}])
        -- print("Termial: cfvs_data: [" .. d .. "][2],  = ", self.cfvs_data_call[{self.term_call_indices[d]}]:size())
      end
    end
    self.cfvs_data[d][1]:copy(self.cfvs_data_fold[{self.term_fold_indices[d]}])

    --correctly set the folded player by mutliplying by -1
    local fold_mutliplier = (self.acting_player[d]*2 - 3)
    self.cfvs_data[d][{1, {}, {}, {}, 1, {}}]:mul(fold_mutliplier)
    self.cfvs_data[d][{1, {}, {}, {}, 2, {}}]:mul(-fold_mutliplier)
  end
end

--- Using the players' reach probabilities, calls the neural net to compute the
-- players' counterfactual values at the depth-limited states of the lookahead.
-- @local
function Lookahead:_compute_terminal_equities_next_street_box()

  assert(self.tree.street ~= constants.streets_count)

  if self.num_pot_sizes == 0 then
    return
  end

  for d=2, self.depth do
    if d > 2 or self.first_call_transition then

      -- if there's only 1 parent, then it should've been an all in, so skip this next_street_box calculation
      if self.ranges_data[d][2]:size(1) > 1 or (d == 2 and self.first_call_transition) or not game_settings.nl then
        local parent_indices = {1, -2}
        if d == 2 then
          parent_indices = {1,1}
        elseif not game_settings.nl then
          parent_indices = {}
        end
        self.next_street_boxes_outputs[{self.indices[d], {}, {}, {}}]:copy(self.ranges_data[d][{2, parent_indices, {}, {}, {}, {}}])
      end
    end
  end

  if self.tree.current_player == 2 then
    self.next_street_boxes_inputs:copy(self.next_street_boxes_outputs)
  else
    self.next_street_boxes_inputs[{{}, {}, 1, {}}]:copy(self.next_street_boxes_outputs[{{}, {}, 2, {}}])
    self.next_street_boxes_inputs[{{}, {}, 2, {}}]:copy(self.next_street_boxes_outputs[{{}, {}, 1, {}}])
  end

  if self.tree.street == 1 then
    self.next_street_boxes:get_value_aux(
      self.next_street_boxes_inputs:view(-1, constants.players_count, game_settings.hand_count),
      self.next_street_boxes_outputs:view(-1, constants.players_count, game_settings.hand_count),
      self.next_board_idx)
  else
    self.next_street_boxes:get_value(
      self.next_street_boxes_inputs:view(-1, constants.players_count, game_settings.hand_count),
      self.next_street_boxes_outputs:view(-1, constants.players_count, game_settings.hand_count))
  end

  --now the neural net outputs for P1 and P2 respectively, so we need to swap the output values if necessary
  if self.tree.current_player == 2 then
    self.next_street_boxes_inputs:copy(self.next_street_boxes_outputs)

    self.next_street_boxes_outputs[{{}, {}, 1, {}}]:copy(self.next_street_boxes_inputs[{{}, {}, 2, {}}])
    self.next_street_boxes_outputs[{{}, {}, 2, {}}]:copy(self.next_street_boxes_inputs[{{}, {}, 1, {}}])
  end

  for d=2, self.depth do
    if d > 2 or self.first_call_transition then
      if self.ranges_data[d][2]:size(1) > 1 or (d == 2 and self.first_call_transition) or not game_settings.nl then
        local parent_indices = {1, -2}
        if d == 2 then
          parent_indices = {1,1}
        elseif not game_settings.nl then
          parent_indices = {}
        end
        self.cfvs_data[d][{2, parent_indices, {}, {}, {}, {}}]:copy(self.next_street_boxes_outputs[{self.indices[d], {}, {}, {}}])
      end
    end
  end
end

--- Gives the average counterfactual values for the opponent during re-solving
-- after a chance event (the betting round changes and more cards are dealt).
--
-- Used during continual re-solving to track opponent cfvs. The lookahead must
-- first be re-solved with @{resolve} or @{resolve_first_node}.
--
-- @param action_index the action taken by the re-solving player at the start
-- of the lookahead
-- @param board a tensor of board cards, updated by the chance event
-- @return a vector of cfvs
function Lookahead:get_chance_action_cfv(action, board)

  local box_outputs = self.next_street_boxes_outputs:view(-1, constants.players_count, game_settings.hand_count)
  local next_street_box = self.next_street_boxes
  local batch_index = self.action_to_index[action]
  assert(batch_index ~= nil)
  local pot_mult = self.next_round_pot_sizes[batch_index]

  if box_outputs == nil then
    assert(false)
  end
  next_street_box:get_value_on_board(board, box_outputs)

  local out = box_outputs[batch_index][self.tree.current_player]
  out:mul(pot_mult)

  return out
end

--- Using the players' reach probabilities, computes their counterfactual
-- values at all terminal states of the lookahead.
--
-- These include terminal states of the game and depth-limited states.
-- @local
function Lookahead:_compute_terminal_equities(iter)

  if self.tree.street ~= constants.streets_count then
    -- v3 热启动，使用神经网络估计前面部分
    if iter > arguments.cfr_skip_iters then
      --v2
      -- compute ranges and pots
      local alltimer = torch.Timer()
      alltimer:reset()
      local timer = torch.Timer()
      timer:reset()
      self:_compute_next_ranges_pots()
      -- print("_compute_next_ranges_pots time:" .. timer:time().real)
      timer:reset()
      -- compute sub game tree
      -- v3 定期更新
      if (iter-1) % arguments.cfr_skip_sub_iters == 0 then
        self.sub_cfvs_data = self:_compute_chance_cfvs()
        if(self.old_cfvs_data == nil) then
          self.old_cfvs_data = self.sub_cfvs_data
        else
          local mse = torch.mean(torch.pow((self.sub_cfvs_data - self.old_cfvs_data), 2))
          print("==========================================================")
          print("==================== MSE: ".. string.format( "%0.10f",mse ) .. " ===================")
          print("==================== MAX: ".. string.format( "%0.10f", torch.max(self.sub_cfvs_data)) .. " ===================")
          print("==================== MIN: ".. string.format( "%0.10f", torch.min(self.sub_cfvs_data)) .. " ===================")
          print("==================== MEAN: ".. string.format( "%0.9f", torch.mean(self.sub_cfvs_data)) .. " ===================")
          print("==========================================================")
          self.old_cfvs_data = self.sub_cfvs_data
        end
      end
      -- print("_compute_chance_cfvs time:" .. timer:time().real)
      timer:reset()
      -- copy to current tree
      self:_compute_without_terminal_equities_next_street_box()
      print("iter: " .. iter .. ", _compute_without_terminal_equities_next_street_box time:" .. alltimer:time().real)
    else
      local timer = torch.Timer()
      timer:reset()
      self:_compute_terminal_equities_next_street_box()
      print("iter: " .. iter .. ", _compute_with_next_street_box time:" .. timer:time().real)
    end
  --  self:_compute_terminal_equities_next_street_box()
  end

  self:_compute_terminal_equities_terminal_equity()
  --multiply by pot scale factor
  for d=2,self.depth do
    self.cfvs_data[d]:cmul(self.pot_size[d])
  end
end

-- v2 计算后续的ranges和pots
function Lookahead:_compute_next_ranges_pots()
  -- pots
  self.next_pots = self.without_nn:get_pot()
  -- ranges
  self.next_ranges_tmp = arguments.Tensor(self.num_pot_sizes, self.batch_size, constants.players_count, game_settings.hand_count):zero()

  for d=2, self.depth do
    if d > 2 or self.first_call_transition then

      -- if there's only 1 parent, then it should've been an all in, so skip this next_street_box calculation
      if self.ranges_data[d][2]:size(1) > 1 or (d == 2 and self.first_call_transition) or not game_settings.nl then
        local parent_indices = {1, -2}
        if d == 2 then
          parent_indices = {1,1}
        elseif not game_settings.nl then
          parent_indices = {}
        end
        self.next_ranges_tmp[{self.indices[d], {}, {}, {}}]:copy(self.ranges_data[d][{2, parent_indices, {}, {}, {}, {}}])
      end
    end
  end

  -- next_range shape --[pot_size_num, batch_size, player, 1326]
  if self.tree.current_player == 2 then
    self.next_ranges_p1 = self.next_ranges_tmp[{{}, {}, 1, {}}]:clone()
    self.next_ranges_p2 = self.next_ranges_tmp[{{}, {}, 2, {}}]:clone()
  else
    self.next_ranges_p1 = self.next_ranges_tmp[{{}, {}, 2, {}}]:clone()
    self.next_ranges_p2 = self.next_ranges_tmp[{{}, {}, 1, {}}]:clone()
  end
  
  -- next_ranges_px shape, [pot_size_num, batch_size, 1326], require [batch_size, 1326]
  -- print("pot: ", self.next_pots:size(1))
  -- print(self.next_pots)
  -- print("ranges: ", self.next_ranges_tmp:size(1))
  assert(self.next_pots:size(1) == self.next_ranges_tmp:size(1))

end

--v2
function Lookahead:_compute_without_terminal_equities_next_street_box()
  for d=2, self.depth do
    if d > 2 or self.first_call_transition then
      if self.ranges_data[d][2]:size(1) > 1 or (d == 2 and self.first_call_transition) or not game_settings.nl then
        local parent_indices = {1, -2}
        if d == 2 then
          parent_indices = {1,1}
        end
        -- v2
        -- self.cfvs_data[d][{2, parent_indices, {}, {}, {}, {}}]:copy(self.sub_cfvs_data[{self.indices[d], {}, {}, {}}])
        -- remove indice from network construct part
        -- print('self.sub_cfvs_data',self.sub_cfvs_data:size())                                             -- [1, 100, 2, 1326]
        -- print('self.cfvs_data',self.cfvs_data:size())
        -- print('self.cfvs_data[d]',self.cfvs_data[d]:size())                                        -- [4, 3, 1, 100, 2, 1326]
        -- print('self.cfvs_data[d][{2, parent_indices, {}, {}, {}, {}}]',self.cfvs_data[d][{2, parent_indices, {}, {}, {}, {}}]:size())      -- [2, 1, 100, 2, 1326]
        self.cfvs_data[d][2][{parent_indices, {}, {}, {}, {}}]:copy(self.sub_cfvs_data[{self.indices[d], {}, {}, {}}])
        -- print("Without Terimal: ", parent_indices)
        -- print("Without Terimal: [" .. d .. "][ parent_indices ][{}, {}, {}, {}] = sub_cfvs_data", self.indices[d])
      end
    end
  end
end


--v2
function Lookahead:_compute_chance_cfvs()
  self.ans = arguments.Tensor(self.num_pot_sizes, self.batch_size, constants.players_count, game_settings.hand_count):zero()
  local size_ = self.next_pots:size(1)

  if self.next_pots[-1][1] == arguments.stack then
    self.ans[size_]:copy(self:resolve_for_allin_chance_cfv())
    size_ = size_ - 1
  end
  for i=1, size_ do
    -- 对于当前的pot i来计算所有可能的棋盘，求平均
    self.ans[i]:copy(self:resolve_for_chance_cfv(i))
  end
  return self.ans:mul(1 / self.next_boards:size(1))
end


require 'DataGeneration.range_generator'
require 'Lookahead.sub_resolving'
--v2
-- 通过调用 lookahead --> subresovling -->  sublookahead来解算以 inner_node 为root的子博弈树
function Lookahead:resolve_for_chance_cfv(pot_idx)
  -- ##################################### New code for chance cfv ##########################################
  local batch_size = arguments.gen_batch_size

  -- 根据之前存储的取出对应id的pot和range
  local pot_size = self.next_pots[pot_idx][1]
  local upper_p1_range = self.next_ranges_p1[pot_idx]
  local upper_p2_range = self.next_ranges_p2[pot_idx]
  local total_values = arguments.Tensor(self.batch_size, constants.players_count, game_settings.hand_count):zero()
  -- 对每一个board计算
  local timer = torch.Timer()
  timer:reset()

  
  for i = 1, self.next_boards:size(1) do
    -- print("==============================================================")
    local board = self.next_boards[i]
    local t = torch.Timer()
    t:reset()
    local p1_range = upper_p1_range:clone()
    local p2_range = upper_p2_range:clone()
    self:update_next_board_ranges(board, p1_range)
    self:update_next_board_ranges(board, p2_range)
    -- print("Update ranges time : " .. t:time().real)
    t:reset()

    -- local te = TerminalEquity()
    local ter = self:deep_copy_te()
    ter:set_board(board)
    -- print("Set Terminal time : " .. t:time().real)

    local resolving = SubResolving(ter)
    local current_node = {}
    local street = 4
    current_node.board = board
    current_node.street = street
    current_node.num_bets = 0
    current_node.current_player = street == 1 and constants.players.P1 or constants.players.P2

    current_node.bets = arguments.Tensor{pot_size, pot_size}

    t:reset()
    resolving:resolve_first_node(current_node, p1_range, p2_range)
    -- print("sub resolving time: " .. t:time().real)
    local root_values = resolving:get_root_cfv_both_players()
    -- print("get_root_cfv_both_players time: " .. t:time().real)
    t:reset()
    root_values:mul(1/pot_size)
    -- print("mul time: " .. t:time().real)
    t:reset()
    total_values:add(root_values)
    -- print("add time: " .. t:time().real)

    -- print("solve pot value: " .. pot_size .. ", board value: " .. card_to_string:cards_to_string(board) .. ", sub total time: " .. timer:time().real)
    timer:reset()

    -- print("==============================================================")
  end

  total_values:mul(1 / self.next_boards:size(1))
  return total_values
  -- ##################################### Old code for chance cfv ##########################################

  -- local batch_size = arguments.gen_batch_size
  -- local te = TerminalEquity()
  -- te:set_board(board)
  -- local range_generator = RangeGenerator()
  -- range_generator:set_board(te, board)

  -- local ranges = arguments.Tensor(constants.players_count, batch_size, 1326)
  -- for player = 1, constants.players_count do
  --   range_generator:generate_range(ranges[player])
  -- end

  -- local min_pot = {100,200,400,2000,6000}
  -- local max_pot = {100,400,2000,6000,18000}
  -- local pot_range = {}
  -- for i = 1,#min_pot do
  --   pot_range[i] = max_pot[i] - min_pot[i]
  -- end
  -- local random_pot_cat = torch.rand(1):mul(#min_pot):add(1):floor()[1]
  -- local random_pot_size = torch.rand(1)[1]

  -- random_pot_size = random_pot_size * pot_range[random_pot_cat]
  -- random_pot_size = random_pot_size + min_pot[random_pot_cat]
  -- random_pot_size = math.floor(random_pot_size)
  -- local pot_size = random_pot_size

  -- local resolving = SubResolving(te)
  -- local current_node = {}
  -- local street = 4
  -- current_node.board = board
  -- current_node.street = street
  -- current_node.num_bets = 0
  -- current_node.current_player = street == 1 and constants.players.P1 or constants.players.P2

  -- current_node.bets = arguments.Tensor{pot_size, pot_size}
  -- local p1_range = ranges[1]
  -- local p2_range = ranges[2]
  -- resolving:resolve_first_node(current_node, p1_range, p2_range)
  -- local root_values = resolving:get_root_cfv_both_players()
  -- root_values:mul(1/pot_size)
  -- return root_values
end

--v2
function Lookahead:update_next_board_ranges(next_board, range)
  local possible_hand_indexes = card_tools:get_possible_hand_indexes(next_board)
  local possible_hands_count = possible_hand_indexes:sum(1)[1]
  local possible_hands_mask = possible_hand_indexes:view(1, -1)
  if not arguments.gpu then
    possible_hands_mask = possible_hands_mask:byte()
  else
    possible_hands_mask = possible_hands_mask:cudaByte()
  end
  range:maskedCopy(possible_hands_mask:expandAs(range), range)
end

-- v2
-- 为了处理pot == stack这种情况， 修改树结构
function Lookahead:resolve_for_allin_chance_cfv()
  local batch_size = arguments.gen_batch_size

  -- 根据之前存储的取出对应id的pot和range
  local pot_size = self.next_pots[-1][1]
  -- allin 调换P2， P1
  local upper_p2_range = self.next_ranges_p1[-1]
  local upper_p1_range = self.next_ranges_p2[-1]
  local total_values = arguments.Tensor(self.batch_size, constants.players_count, game_settings.hand_count):zero()
  -- 对每一个board计算
  local timer = torch.Timer()
  timer:reset()
  local te = TerminalEquity()
  for i = 1, self.next_boards:size(1) do
    -- print("==============================================================")
    local board = self.next_boards[i]
    local t = torch.Timer()
    t:reset()
    local p1_range = upper_p1_range:clone()
    local p2_range = upper_p2_range:clone()
    self:update_next_board_ranges(board, p1_range)
    self:update_next_board_ranges(board, p2_range)
    -- print("Update ranges time : " .. t:time().real)
    t:reset()

    -- local te = TerminalEquity()
    local ter = self:deep_copy_te()
    ter:set_board(board)
    -- print("Set Terminal time : " .. t:time().real)

    local resolving = SubResolving(ter)
    local current_node = {}
    local street = 4
    current_node.board = board
    current_node.street = street
    current_node.num_bets = 0
    current_node.current_player = constants.players.P1

    current_node.bets = arguments.Tensor{pot_size, pot_size}

    t:reset()
    resolving:resolve_first_node(current_node, p1_range, p2_range)
    -- print("sub resolving time: " .. t:time().real)
    t:reset()
    local root_values = resolving:get_root_cfv_both_players()
    -- print("get_root_cfv_both_players time: " .. t:time().real)
    t:reset()
    root_values:mul(1/pot_size)
    -- print("mul time: " .. t:time().real)
    t:reset()
    total_values:add(root_values)
    -- print("add time: " .. t:time().real)

    -- print("solve pot value: " .. pot_size .. ", board value: " .. card_to_string:cards_to_string(board) .. ", sub total time: " .. timer:time().real)
    timer:reset()

    -- print("==============================================================")
  end

  total_values:mul(1 / self.next_boards:size(1))
  return total_values
end

--- Using the players' reach probabilities and terminal counterfactual
-- values, computes their cfvs at all states of the lookahead.
-- @local
function Lookahead:_compute_cfvs()
  for d=self.depth,2,-1 do
    local gp_layer_terminal_actions_count = self.terminal_actions_count[d-2]
    local ggp_layer_nonallin_bets_count = self.nonallinbets_count[d-3]

    self.cfvs_data[d][{{}, {}, {}, {}, {1}, {}}]:cmul(self.empty_action_mask[d])
    self.cfvs_data[d][{{}, {}, {}, {}, {2}, {}}]:cmul(self.empty_action_mask[d])

    self.placeholder_data[d]:copy(self.cfvs_data[d])

    --player indexing is swapped for cfvs
    self.placeholder_data[d][{{}, {}, {}, {}, self.acting_player[d], {}}]:cmul(self.current_strategy_data[d])

    torch.sum(self.regrets_sum[d], self.placeholder_data[d], 1)

    --use a swap placeholder to change {{1,2,3}, {4,5,6}} into {{1,2}, {3,4}, {5,6}}
    local swap = self.swap_data[d-1]
    swap:copy(self.regrets_sum[d])

    self.cfvs_data[d-1][{{gp_layer_terminal_actions_count+1, -1}, {1, ggp_layer_nonallin_bets_count}, {}, {}, {}, {}}]:copy(swap:transpose(2,3))
  end

end

--- Updates the players' average counterfactual values with their cfvs from the
-- current iteration.
-- @param iter the current iteration number of re-solving
-- @local
function Lookahead:_compute_cumulate_average_cfvs(iter)
  if iter > arguments.cfr_skip_iters then
    self.average_cfvs_data[1]:add(self.cfvs_data[1])

    self.average_cfvs_data[2]:add(self.cfvs_data[2])
  end
end

--- Normalizes the players' average strategies.
--
-- Used at the end of re-solving so that we can track un-normalized average
-- strategies, which are simpler to compute.
-- @local
function Lookahead:_compute_normalize_average_strategies()

  --using regrets_sum as a placeholder container
  local player_avg_strategy = self.average_strategies_data[2]
  local player_avg_strategy_sum = self.regrets_sum[2]


  torch.sum(player_avg_strategy_sum, player_avg_strategy, 1)
  player_avg_strategy:cdiv(player_avg_strategy_sum:expandAs(player_avg_strategy))

  --if the strategy is 'empty' (zero reach), strategy does not matter but we need to make sure
  --it sums to one -> now we set to always fold
  player_avg_strategy[1][player_avg_strategy[1]:ne(player_avg_strategy[1])] = 1
  player_avg_strategy[player_avg_strategy:ne(player_avg_strategy)] = 0
end

--- Normalizes the players' average counterfactual values.
--
-- Used at the end of re-solving so that we can track un-normalized average
-- cfvs, which are simpler to compute.
-- @local
function Lookahead:_compute_normalize_average_cfvs()
  self.average_cfvs_data[1]:div(arguments.cfr_iters - arguments.cfr_skip_iters)
end

--- Using the players' counterfactual values, updates their total regrets
-- for every state in the lookahead.
-- @local
function Lookahead:_compute_regrets()
  for d=self.depth,2,-1 do
    local gp_layer_terminal_actions_count = self.terminal_actions_count[d-2]
    local gp_layer_bets_count = self.bets_count[d-2]
    local ggp_layer_nonallin_bets_count = self.nonallinbets_count[d-3]

    local current_regrets = self.current_regrets_data[d]
    current_regrets:copy(self.cfvs_data[d][{{}, {}, {}, {}, self.acting_player[d], {}}])

    local next_level_cfvs = self.cfvs_data[d-1]

    local parent_inner_nodes = self.inner_nodes_p1[d-1]
    parent_inner_nodes:copy(next_level_cfvs[{{gp_layer_terminal_actions_count+1, -1}, {1, ggp_layer_nonallin_bets_count}, {}, {}, self.acting_player[d], {}}]:transpose(2,3))
    parent_inner_nodes = parent_inner_nodes:view(1, gp_layer_bets_count, -1, self.batch_size, game_settings.hand_count)
    parent_inner_nodes = parent_inner_nodes:expandAs(current_regrets)

    current_regrets:csub(parent_inner_nodes)

    self.regrets_data[d]:add(self.regrets_data[d], current_regrets)

    --(CFR+)
    self.regrets_data[d]:clamp(0,  tools:max_number())
  end
end


--- Gets the results of re-solving the lookahead.
--
-- The lookahead must first be re-solved with @{resolve} or
-- @{resolve_first_node}.
--
-- @return a table containing the fields:
--
-- * `strategy`: an AxK tensor containing the re-solve player's strategy at the
-- root of the lookahead, where A is the number of actions and K is the range size
--
-- * `achieved_cfvs`: a vector of the opponent's average counterfactual values at the
-- root of the lookahead
--
-- * `children_cfvs`: an AxK tensor of opponent average counterfactual values after
-- each action that the re-solve player can take at the root of the lookahead
function Lookahead:get_results()
  local out = {}

  local actions_count = self.average_strategies_data[2]:size(1)

  --1.0 average strategy
  --[actions x range]
  --lookahead already computes the averate strategy we just convert the dimensions
  out.strategy = self.average_strategies_data[2]:view(-1, self.batch_size, game_settings.hand_count):clone()

  --2.0 achieved opponent's CFVs at the starting node
  out.achieved_cfvs = self.average_cfvs_data[1]:view(self.batch_size, constants.players_count, game_settings.hand_count)[{{},1,{}}]:clone()

  --3.0 CFVs for the acting player only when resolving first node
  if self.reconstruction_opponent_cfvs then
    out.root_cfvs = nil
  else
    out.root_cfvs = self.average_cfvs_data[1]:view(self.batch_size, constants.players_count, game_settings.hand_count)[{{},2,{}}]:clone()

    --swap cfvs indexing
    out.root_cfvs_both_players = self.average_cfvs_data[1]:view(self.batch_size, constants.players_count, game_settings.hand_count):clone()
    out.root_cfvs_both_players[{{},2,{}}]:copy(self.average_cfvs_data[1]:view(self.batch_size, constants.players_count, game_settings.hand_count)[{{},1,{}}])
    out.root_cfvs_both_players[{{},1,{}}]:copy(self.average_cfvs_data[1]:view(self.batch_size, constants.players_count, game_settings.hand_count)[{{},2,{}}])
  end

  --4.0 children CFVs
  --[actions x range]
  out.children_cfvs = self.average_cfvs_data[2][{{}, {}, {}, {}, 1, {}}]:clone():view(-1, game_settings.hand_count)

  --IMPORTANT divide average CFVs by average strategy in here
  local scaler = self.average_strategies_data[2]:view(-1, self.batch_size, game_settings.hand_count):clone()


  local range_mul = self.ranges_data[1][{{}, {}, {}, {}, 1, {}}]:clone():view(1, self.batch_size, game_settings.hand_count):clone()
  range_mul = range_mul:expandAs(scaler)

  scaler = scaler:cmul(range_mul)
  scaler = scaler:sum(3):expandAs(range_mul):clone()
  scaler = scaler:mul(arguments.cfr_iters - arguments.cfr_skip_iters)

  out.children_cfvs:cdiv(scaler)

  assert(out.children_cfvs)
  assert(out.strategy)
  assert(out.achieved_cfvs)

  return out
end

--- Generates the opponent's range for the current re-solve iteration using
-- the @{cfrd_gadget|CFRDGadget}.
-- @param iteration the current iteration number of re-solving
-- @local
function Lookahead:_set_opponent_starting_range(iteration)
  if self.reconstruction_opponent_cfvs then
    --note that CFVs indexing is swapped, thus the CFVs for the reconstruction player are for player '1'
    local opponent_range = self.reconstruction_gadget:compute_opponent_range(self.cfvs_data[1][{{}, {}, {}, {}, 1, {}}], iteration)
    self.ranges_data[1][{{}, {}, {}, {}, 2, {}}]:copy(opponent_range)
  end
end

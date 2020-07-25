local arguments = require 'Settings.arguments'
local constants = require 'Settings.constants'
local card_to_string = require 'Game.card_to_string_conversion'
require 'Tree.tree_builder'


local builder = PokerTreeBuilder()

local params = {}

params.root_node = {}
params.root_node.board = card_to_string:string_to_board('')
params.root_node.street = 4
params.root_node.current_player = constants.players.P1
params.root_node.bets = arguments.Tensor{20000, 20000}
params.root_node.num_bets = 0
params.limit_to_street = true

local tree = builder:build_tree(params)
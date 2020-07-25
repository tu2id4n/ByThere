--- Uses the neural net to estimate value at the end of the first betting round.
-- @classmod next_round_value

require 'torch'
require 'math'

local arguments = require 'Settings.arguments'
local game_settings = require 'Settings.game_settings'

local WithoutNN = torch.class('WithoutNN')

function WithoutNN:__init()
end

-- v2
-- modified from Nn/next_round_value.lua
function WithoutNN:start_computation(pot_sizes, batch_size)
    self.pot_sizes = pot_sizes:view(-1, 1):clone()
end

function WithoutNN:get_pot()
    return self.pot_sizes:clone()
end

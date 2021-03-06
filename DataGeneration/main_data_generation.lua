--- Script that generates training and validation files.
-- @see data_generation
-- @script main_data_generation
local filename = os.time()

if #arg == 0 then
  print("Please specify the street. 1 = preflop, 4 = river")
  return
end

local arguments = require 'Settings.arguments'
local data_generation = require 'DataGeneration.data_generation'
print(os.time() - filename)
data_generation:generate_data(arguments.train_data_count, filename, tonumber(arg[1]))

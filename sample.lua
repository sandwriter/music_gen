
--[[

   This file samples characters from a trained model

   Code is based on implementation in 
   https://github.com/oxford-cs-ml-2015/practical6

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.ConvertToTwoDimension'
require 'util.misc'

nngraph.setDebug(true)

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-primetext',"",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-length',1000,'number of quarter note to sample')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- gated print: simple utility function wrapping a print
function gprint(str)
   if opt.verbose == 1 then print(str) end
end

torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
   gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the rnn state to all zeros
gprint('creating an LSTM...')
local current_state
local num_layers = checkpoint.opt.num_layers
current_state = {}
for L = 1,checkpoint.opt.num_layers do
   -- c and h for all layers
   local h_init = torch.zeros(1, checkpoint.opt.rnn_size)
   table.insert(current_state, h_init:clone())
   table.insert(current_state, h_init:clone())
end
state_size = #current_state

-- do a few seeded timesteps
gprint('missing seed text, using null for the first sound.')
gprint('--------------------------')
prediction = torch.DoubleTensor{-7.93976819724, -7.33851285508, -8.28845703503, -5.86963025143, -3.80446572541, -3.4923272897, -3.78412307061, -2.31533176223, -2.80574743898, -3.15260828984, -1.6581434454, -1.31461229068, -1.00237280568, -0.425871653287, -0.047943520908, 0.437183076234, 1.36561563117, 1.83967332492, 1.49233534565, 2.12384874082, 2.57148330458, 3.14590471347, 3.13023254119, 3.10024031642, 3.59681178615, 4.12782545459}
prediction = prediction:resize(1, 26)

-- start sampling/argmaxing
for i=1, opt.length do

   -- -- log probabilities from the previous timestep
   -- if opt.sample == 0 then
   --     -- use argmax
   --     local _, prev_char_ = prediction:max(2)
   --     prev_char = prev_char_:resize(1)
   -- else
   --     -- use sampling
   --     prediction:div(opt.temperature) -- scale by temperature
   --     local probs = torch.exp(prediction):squeeze()
   --     probs:div(torch.sum(probs)) -- renormalize so probs sum to one
   --     prev_char = torch.multinomial(probs:float(), 1):resize(1):float()
   -- end

   -- prev_sound = prediction + torch.randn(1, 26)
   prev_sound = prediction

   -- forward the rnn for next character
   local lst = protos.rnn:forward{prev_sound, prev_sound, unpack(current_state)}
   current_state = {}
   for i=1,state_size do
      table.insert(current_state, lst[i])
      -- print('list ', i, lst[i])
   end

   -- predicte using mixture density distribution

   local pi = lst[#lst-3]
   local mu = lst[#lst-2]
   local sigma = lst[#lst-1]

   local dimension = mu:size(3)

   local current = -1
   local argmax = 0
   for i=1, pi:size(2) do
      local tmp = pi[1][i] * math.pow(sigma[1][i], -dimension)
      if tmp > current then
	 argmax = i
      end
   end

   local tmp_mu = mu:select(2, argmax)
   prediction_array = {}
   for i=1, dimension do
      prediction_array[i] = torch.normal(tmp_mu[1][i], sigma[1][argmax])
   end

   prediction = torch.Tensor(prediction_array):resize(1, dimension)
   
   print(prediction)
   print('mu:', tmp_mu, 'sigma:', sigma[1][argmax])
end
io.write('\n') io.flush()


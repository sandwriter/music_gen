require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'audio'

require 'util.OneHot'
require 'util.ConvertToTwoDimension'
require 'util.misc'

local AudioLoader = require 'util.AudioLoader'
local model_utils = require 'util.model_utils'

nngraph.setDebug(true)
-- local MDN = require 'model.mdn'
local LSTM = require 'model.LSTM'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a one-frequency audio model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/audio','data directory. Should contain the file audio.mp3 with input data')
-- model params
cmd:option('-window_size', 27552, 'window size for spectrogram.')
cmd:option('-stride', 27552, 'stride should be the same number as window_size.')
cmd:option('-rnn_size', 256, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-num_component', 5, 'number of Gaussian compoents in the mixture density model')
cmd:option('-model', 'lstm', 'for now only lstm is supported. keep fixed')
-- optimization
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay',0.95,'learning rate decay')
cmd:option('-learning_rate_decay_after',900,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0,'dropout to use just before classifier. 0 = no dropout')
cmd:option('-seq_length',500,'number of timesteps to unroll for')
cmd:option('-max_epochs', 1000,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at')
cmd:option('-train_frac',1.0,'fraction of data that goes into train set')
cmd:option('-val_frac',0.0,'fraction of data that goes into validation set')
-- note: test_frac will be computed as (1 - train_frac - val_frac)
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
-- GPU/CPU
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
-- train / val / test split for data, in fractions
local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
local split_sizes = {opt.train_frac, opt.val_frac, test_frac}

-- create the data loader class
local loader = AudioLoader.create(opt.data_dir, opt.seq_length, split_sizes, opt.window_size, opt.stride)
-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- define the model: prototypes for one timestep, then clone them in time
protos = {}
print('creating an LSTM with ' .. opt.num_layers .. ' layers')
-- input size is 2. i.e, (freqency, magnitude)
-- TODO: use top 5 freq or look at MFCC.
protos.rnn = LSTM.lstm(26, opt.rnn_size, opt.num_layers, opt.dropout, opt.num_component)
-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
   local h_init = torch.zeros(1, opt.rnn_size)
   table.insert(init_state, h_init:clone())
   table.insert(init_state, h_init:clone())
end
-- training criterion (negative log likelihood)
-- protos.criterion = nn.MDNCriterion()

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)

-- initialization
params:uniform(-0.08, 0.08) -- small numbers uniform

print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
   print('cloning ' .. name)
   clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
   print('evaluating loss over split index ' .. split_index)
   local n = loader.split_sizes[split_index]
   if max_batches ~= nil then n = math.min(max_batches, n) end

   loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
   local loss = 0
   local rnn_state = {[0] = init_state}
   
   for i = 1,n do -- iterate over batches in the split
      -- fetch a batch
      local x, y = loader:next_batch(split_index)
      -- forward pass
      for t=1,opt.seq_length do
	 clones.rnn[t]:evaluate() -- for dropout proper functioning
	 local lst = clones.rnn[t]:forward{x[{{}, t}], unpack(rnn_state[t-1])}
	 rnn_state[t] = {}
	 for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
	 prediction = lst[#lst] 
	 loss = loss + clones.criterion[t]:forward(prediction, y[{{}, t}])
      end
      -- carry over lstm state
      rnn_state[0] = rnn_state[#rnn_state]
      print(i .. '/' .. n .. '...')
   end

   loss = loss / opt.seq_length / n
   return loss
end

-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)
function feval(x)
   if x ~= params then
      params:copy(x)
   end
   grad_params:zero()

   ------------------ get minibatch -------------------
   local x, y = loader:next_batch(1)

   ------------------- forward pass -------------------
   local rnn_state = {[0] = init_state_global}
   local predictions = {}           -- softmax outputs
   local loss = 0
   for t=1,opt.seq_length do
      clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
      local lst = clones.rnn[t]:forward{x[{{}, t}], y[{{}, t}], unpack(rnn_state[t-1])}
      rnn_state[t] = {}
      for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output

      pi = lst[#lst-3]
      mu = lst[#lst-2]
      sigma = lst[#lst-1]
      z5 = lst[#lst]
      z5_value = torch.mean(z5)

      if z5_value ~= z5_value or z5_value == 1/0 then
	 print('z5_value:', z5_value, 'x:', x[{{}, t}], 'y:', y[{{}, t}], 'pi:', pi, 'mu:', mu, 'sigma:', sigma, 'params:', params, 'grad_params:', grad_params)
	 os.exit()
      end

      -- print('z5:', torch.mean(z5), 'x:', x[{{}, t}], 'y:', y[{{}, t}], 'pi:', pi, 'mu:', mu, 'sigma:', sigma)

      loss = loss + z5_value
      -- predictions[t] = lst[#lst] -- last element is the prediction
      -- loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
   end


   
   loss = loss / opt.seq_length
   ------------------ backward pass -------------------
   -- initialize gradient at time t to be zeros (there's no influence from future)
   local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
   for t=opt.seq_length,1,-1 do
      -- backprop through loss, and softmax/linear
      -- local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
      local dpi_t = torch.zeros(pi:size())
      local dmu_t = torch.zeros(mu:size())      
      local dsigma_t = torch.zeros(sigma:size())
      local dz5_t = torch.ones(z5:size())
      table.insert(drnn_state[t], dpi_t)
      table.insert(drnn_state[t], dmu_t)
      table.insert(drnn_state[t], dsigma_t)      
      table.insert(drnn_state[t], dz5_t)
      local dlst = clones.rnn[t]:backward({x[{{}, t}], y[{{}, t}], unpack(rnn_state[t-1])}, drnn_state[t])
      drnn_state[t-1] = {}
      for k,v in pairs(dlst) do
	 if k > 2 then -- k == 1 is gradient on x, which we dont need
	    -- note we do k-1 because first item is dembeddings, and then follow the 
	    -- derivatives of the state, starting at index 2. I know...
	    drnn_state[t-1][k-2] = v
	 end
      end
      -- print('time step', t)
      -- print('params', params)
      -- print('grad_params', grad_params)
   end
   ------------------------ misc ----------------------
   -- transfer final state to initial state (BPTT)
   init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
   -- clip gradient element-wise
   grad_params:clamp(-opt.grad_clip, opt.grad_clip)
   return loss, grad_params
end

-- start optimization here
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil
for i = 1, iterations do
   local epoch = i / loader.ntrain

   local timer = torch.Timer()
   local _, loss = optim.rmsprop(feval, params, optim_state)
   local time = timer:time().real

   local train_loss = loss[1] -- the loss is inside a list, pop it
   train_losses[i] = train_loss

   -- exponential learning rate decay
   if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
      if epoch >= opt.learning_rate_decay_after then
	 local decay_factor = opt.learning_rate_decay
	 optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
	 print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
      end
   end

   -- every now and then or on last iteration
   if i % opt.eval_val_every == 0 or i == iterations then
      -- evaluate loss on validation data
      local val_loss = eval_split(2) -- 2 = validation
      val_losses[i] = val_loss

      local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
      print('saving checkpoint to ' .. savefile)
      local checkpoint = {}
      checkpoint.protos = protos
      checkpoint.opt = opt
      checkpoint.train_losses = train_losses
      checkpoint.val_loss = val_loss
      checkpoint.val_losses = val_losses
      checkpoint.i = i
      checkpoint.epoch = epoch
      torch.save(savefile, checkpoint)
   end

   if i % opt.print_every == 0 then
      print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
   end
   
   if i % 10 == 0 then collectgarbage() end

   -- handle early stopping if things are going really bad
   -- if loss0 == nil then loss0 = loss[1] end
   -- if loss[1] > loss0 * 3 then
   --    print('loss is exploding, aborting.')
   --    break -- halt
   -- end
end



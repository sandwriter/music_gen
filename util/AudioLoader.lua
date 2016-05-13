-- Load audio file, extract only one channel.
-- Not data should be used for test/validation purpose at this point.

local AudioLoader = {}

AudioLoader.__index = AudioLoader

function AudioLoader.create(data_dir, seq_length, split_fractions, window_size, stride)
   -- split_fractions is e.g. {0.9, 0.05, 0.05}

   local self = {}
   setmetatable(self, AudioLoader)

   local input_file = path.join(data_dir, 'data.mp3')
   local tensor_file = path.join(data_dir, 'data.t7')

   -- construct a tensor with all the data
   if not path.exists(tensor_file) then
      print('one-time setup: preprocessing input file ' .. input_file .. '...')
      -- Turn audio file into top K spectrogram tensor
      AudioLoader.codec_to_tensor(input_file, tensor_file, window_size, stride)
   end

   print('loading data files...')
   local data = torch.load(tensor_file)

   -- cut off the end so that it divides evenly
   local len = data:size(2)
   if len % seq_length ~= 0 then
      print('cutting off end of data so that the batches/sequences divide evenly')
      data = data:sub(1, -1, 1, seq_length * math.floor(len / seq_length))
   end

   print('cutted off data size:', data:size())

   -- self.batches is a table of tensors
   print('reshaping tensor...')
   self.seq_length = seq_length

   local ydata = data:clone()
   ydata:sub(1, -1, 1,-2):copy(data:sub(1, -1, 2,-1))
   ydata[{{},-1}] = data[{{},1}]
   self.x_batches = data:split(seq_length, 2)
   self.nbatches = #self.x_batches
   self.y_batches = ydata:split(seq_length, 2)
   assert(#self.x_batches == #self.y_batches)

   -- lets try to be helpful here
   if self.nbatches < 50 then
      print('WARNING: less than 50 batches in the data in total? Looks like very small dataset. You probably want to use smaller batch_size and/or seq_length.')
   end

   -- perform safety checks on split_fractions
   assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
   assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
   assert(split_fractions[3] >= 0 and split_fractions[3] <= 1, 'bad split fraction ' .. split_fractions[3] .. ' for test, not between 0 and 1')
   if split_fractions[3] == 0 then 
      -- catch a common special case where the user might not want a test set
      self.ntrain = math.floor(self.nbatches * split_fractions[1])
      self.nval = self.nbatches - self.ntrain
      self.ntest = 0
   else
      -- divide data to train/val and allocate rest to test
      self.ntrain = math.floor(self.nbatches * split_fractions[1])
      self.nval = math.floor(self.nbatches * split_fractions[2])
      self.ntest = self.nbatches - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)
   end

   self.split_sizes = {self.ntrain, self.nval, self.ntest}
   self.batch_ix = {0,0,0}

   print(string.format('data load done. Number of data batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))
   collectgarbage()
   return self
end

function AudioLoader:reset_batch_pointer(split_index, batch_index)
   batch_index = batch_index or 0
   self.batch_ix[split_index] = batch_index
end

function AudioLoader:next_batch(split_index)
   if self.split_sizes[split_index] == 0 then
      -- perform a check here to make sure the user isn't screwing something up
      local split_names = {'train', 'val', 'test'}
      print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
      os.exit() -- crash violently
   end
   -- split_index is integer: 1 = train, 2 = val, 3 = test
   self.batch_ix[split_index] = self.batch_ix[split_index] + 1
   if self.batch_ix[split_index] > self.split_sizes[split_index] then
      self.batch_ix[split_index] = 1 -- cycle around to beginning
   end
   -- pull out the correct next batch
   local ix = self.batch_ix[split_index]
   if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
   if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + val
   return self.x_batches[ix], self.y_batches[ix]
end

-- *** STATIC method ***
-- This function load and extract the 1st channel of the audio file, and then calculate the spectrogram of the audio, and gather the top 1 (freq, magnitude).
function AudioLoader.codec_to_tensor(in_audiofile, out_tensorfile, window_size, stride)
   local timer = torch.Timer()

   print('loading audio file...')
   music = audio.load(in_audiofile)
   print('music size', music:size())

   sampling_freq = 44100.0

   -- 1st channel of the audio.
   music_l = music[1]:clone():resize(1, music:size(2))
   print('music_l size', music_l:size())

   -- calculate spectrogram.
   -- use the window as the seq_len, and the striding using seq_len.
   spect = audio.spectrogram(music_l, window_size, 'hann', stride)

   print('spect size', spect:size())

   -- top 1 magnitude and index.
   magnitude, index = torch.max(spect, 1)
   index = index:double()

   -- convert from index to freq.
   -- Normalize by an extra factor of 1000
   freq = (-index + window_size/2.0 + 1.0) * sampling_freq / window_size
   -- print('freq', freq, 'magn', magnitude)

   -- mock freq
   freq = torch.Tensor{0,1,2,3,4,5,4,3,2,1,0,1,2,3,4,5,4,3,2,1,0,1,2,3,4,5,4,3,2,1,0,1,2,3,4,5,4,3,2,1,0,1,2,3,4,5,4,3}
   magnitude = torch.Tensor{1,2,3,4,5,4,3,2,1,0,1,2,3,4,5,4,3,2,1,0,1,2,3,4,5,4,3,2,1,0,1,2,3,4,5,4,3,2,1,0,1,2,3,4,5,4,3,2}

   data = torch.DoubleTensor(48502, 26)
   s = data:storage()

   local file = io.open("/Users/wenjie/Workspace/lstm_k/data/audio/fbank_feat.txt", "r");
   local content = file:read("*a")

   i = 0
   for c in string.gmatch(content, "%S+") do
      i = i + 1
      s[i] = tonumber(c)
   end

   data = data:transpose(1, 2)
   print('data:', data)


   -- print('freq size', freq:size())

   -- -- print('frequency\n', freq)
   -- -- concatenate two tensor into one.
   -- -- data = torch.Tensor(2, index:size(2))
   -- data = torch.Tensor(2, index:size(2))
   -- data[1]:copy(freq)
   -- data[2]:copy(magnitude)

   print('data size', data:size())

   print('saving ' .. out_tensorfile)
   torch.save(out_tensorfile, data)
end

return AudioLoader



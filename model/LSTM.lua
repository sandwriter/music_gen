
local LSTM = {}
function LSTM.lstm(input_size, rnn_size, n, dropout, num_component)
   dropout = dropout or 0 

   -- there will be 2*n+1 inputs
   local inputs = {}
   table.insert(inputs, nn.Identity()()) -- x
   table.insert(inputs, nn.Identity()()) --y
   for L = 1,n do
      table.insert(inputs, nn.Identity()()) -- prev_c[L]
      table.insert(inputs, nn.Identity()()) -- prev_h[L]
   end

   local x, y, input_size_L
   local outputs = {}
   for L = 1,n do
      -- c,h from previos timesteps
      local prev_h = inputs[L*2+2]
      local prev_c = inputs[L*2+1]
      -- the input to this layer
      if L == 1 then
	 x = ConvertToTwoDimension(input_size)(inputs[1])
	 y = ConvertToTwoDimension(input_size)(inputs[2])
	 -- x = OneHot(input_size)(inputs[1])
	 input_size_L = input_size
      else 
	 x = outputs[(L-1)*2] 
	 if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
	 input_size_L = rnn_size
      end
      print('level:', L, '\t input size:', input_size_L)
      -- evaluate the input sums at once for efficiency
      -- W_xi*x_t, W_xf*x_t, W_xc*x_t, W_xo*x_t
      local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x)
      -- W_hi*h_t-1, ...
      local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
      local all_input_sums = nn.CAddTable()({i2h, h2h})
      -- decode the gates
      local sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(all_input_sums)
      sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
      local in_gate = nn.Narrow(2, 1, rnn_size)(sigmoid_chunk)
      local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(sigmoid_chunk)
      local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(sigmoid_chunk)
      -- decode the write inputs
      local in_transform = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(all_input_sums)
      in_transform = nn.Tanh()(in_transform)
      -- perform the LSTM update
      local next_c           = nn.CAddTable()({
	    nn.CMulTable()({forget_gate, prev_c}),
	    nn.CMulTable()({in_gate,     in_transform})
					     })
      -- gated cells form the output
      local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
      
      table.insert(outputs, next_c)
      table.insert(outputs, next_h)
   end

   -- set up the decoder
   local top_h = outputs[#outputs]
   if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end

      -- mu_ij for each component, dimension.
   tilt_mu = nn.Linear(rnn_size, num_component*input_size)(top_h)
   -- sigma_ij for each component.
   tilt_sigma = nn.Linear(rnn_size, num_component)(top_h)
   -- pi_i for each component.
   tilt_pi = nn.Linear(rnn_size, num_component)(top_h)

   -- Converted.
   pi = nn.SoftMax()(tilt_pi)
   mu = nn.Reshape(num_component, input_size)(tilt_mu)
   sigma = nn.Exp()(tilt_sigma)

   l = {}

   for i = 1, num_component do
      z1 = nn.Select(2, i)(mu)
      z2 = nn.MulConstant(-1)(z1)
      d = nn.CAddTable()({z2, y})
      sq = nn.Square()(d)
      s = nn.Sum(2)(sq)
      s = nn.MulConstant(-0.5)(s)
      -- Try to make first few iterations less likely to be nan.
      sigma = nn.MulConstant(10)(sigma)
      sigma_select = nn.Select(2, i)(sigma)
      sigma_sq_inv = nn.Power(-2)(sigma_select)
      pi_select = nn.Select(2, i)(pi)
      mm = nn.CMulTable()({s, sigma_sq_inv})
      e = nn.Exp()(mm)
      sigma_mm = nn.Power(-input_size)(sigma_select)
      r = nn.CMulTable()({e, sigma_mm, pi_select})
      r = nn.MulConstant(math.pow((2*math.pi), -0.5*input_size))(r)
      l[#l + 1] = r
   end

   z3 = nn.CAddTable()(l)
   -- z3 = nn.AddConstant(0.00000000000001)(z3)
   z4 = nn.Log()(z3)
   z5 = nn.MulConstant(-1)(z4)

   -- local proj = nn.Linear(rnn_size, input_size)(top_h)
   table.insert(outputs, pi)
   table.insert(outputs, mu)
   table.insert(outputs, sigma)
   table.insert(outputs, z5)
   -- local logsoft = nn.LogSoftMax()(proj)
   -- table.insert(outputs, logsoft)

   return nn.gModule(inputs, outputs)
end

return LSTM


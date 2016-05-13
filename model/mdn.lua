local MDN = {}

function MDN.mdn(hidden_size, num_component, dimension)
   hidden = nn.Identity()()
   -- mu_ij for each component, dimension.
   tilt_mu = nn.Linear(hidden_size, num_component*dimension)(hidden)
   -- sigma_ij for each component.
   tilt_sigma = nn.Linear(hidden_size, num_component)(hidden)
   -- pi_i for each component.
   tilt_pi = nn.Linear(hidden_size, num_component)(hidden)

   -- Converted.
   pi = nn.SoftMax()(tilt_pi)
   mu = nn.Reshape(dimension, num_component)(tilt_mu)
   sigma = nn.Exp()(tilt_sigma)

   target = nn.Identity()()

   l = {}

   for i = 1, num_component do
      z1 = nn.Select(2, i)(mu)
      z2 = nn.MulConstant(-1)(z1)
      d = nn.CAddTable()({z2, target})
      sq = nn.Square()(d)
      s = nn.Sum(1)(sq)
      s = nn.MulConstant(-0.5)(s)
      sigma_select = nn.Select(1, i)(sigma)
      sigma_sq_inv = nn.Power(-2)(sigma_select)
      pi_select = nn.Select(1, i)(pi)
      mm = nn.CmulTable()({s, sigma_sq_inv})
      e = nn.Exp()(mm)
      sigma_mm = nn.Power(-dimension)(sigma_select)
      r = nn.CMulTable()({e, sigma_mm, pi_select})
      r = nn.MulConstant(math.pow((2*math.pi), -0.5*dimension))(r)
      l[#l + 1] = r
   end

   z3 = nn.CAddTable()(l)
   z4 = nn.Log()(z3)
   z5 = nn.MulConstant(-1)(z4)

   return nn.gModule({hidden, target}, {z5, pi, mu, sigma})
end

return MDN
   
   


local ConvertToTwoDimension, parent = torch.class('ConvertToTwoDimension', 'nn.Module')

function ConvertToTwoDimension:__init(outputSize)
   parent.__init(self)
   self.outputSize = outputSize
end

function ConvertToTwoDimension:updateOutput(input)
   self.output:resize(1, self.outputSize):copy(input)
   return self.output
end


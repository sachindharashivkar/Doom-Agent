package.path = package.path .. ";./vizdoom/?.lua"
require "vizdoom.init"
image = require "image"
threads = require "threads"
require "nn"
require "dpnn"
require "rnn"

local config_file_path = "../../examples/config/health_gathering.cfg"
local T = 0
local TMAX = 10000
local discount_factor = 0.99
local learning_rate = 0.0007
local epoch = 0
local num_epochs = 10
local row = 0.99
local nthread = 16

local function initialize_vizdoom(config_file_path)
	local game = vizdoom.DoomGame()
	game:loadConfig(config_file_path)
	game:setWindowVisible(false)
	game:setMode(vizdoom.Mode.PLAYER)
	game:setScreenResolution(vizdoom.ScreenResolution.RES_160X120)
	game:setScreenFormat(vizdoom.ScreenFormat.CRCGCB)
	game:init()
	print("Doom Initialized")
	return game
end

--Creation of Networks
local net = nn.Sequential()
local vnet = nn.Sequential()	
local net1 = nn.Sequential()
net1:add(nn.SpatialConvolution(3,6,8,8,4,4))
net1:add(nn.ReLU())
net1:add(nn.SpatialConvolution(6,10,4,4,2,2))
net1:add(nn.ReLU())
net1:add(nn.View(1, 360))
net1:add(nn.FastLSTM(360,60,20))
-- For A3C-FF,
--net1:add(nn.Linear(360,60,20))
net:add(net1)
net:add(nn.Linear(60,3))
net:add(nn.SoftMax())
net:add(nn.CategoricalEntropy(0.005))
net:add(nn.ReinforceCategorical(true))
vnet:add(net1)
vnet:add(nn.Linear(60,1))
vnet:share(net,"net1")
torch.save("net.dat", net )
torch.save("vnet.dat", vnet)
print ("Networks created")

local w, g = net:parameters()
local vw,  vg = vnet:parameters()
rg = {}
rvg = {}
for k, v in pairs(g) do
	rg[k] = g[k]:clone():fill(0)
end
for k, v in pairs(vg) do
	rvg[k] = vg[k]:clone():fill(0)
end


local pool = threads.Threads(
   	nthread,
   	function(threadid)
		require "nn"
		require "dpnn"
		require "rnn"	
		require "vizdoom.init"	
		require "image" 
   	end ,  
	function(threadid)
		game = initialize_vizdoom(config_file_path)
		net2 = net:clone()
		vnet2 = vnet:clone()
	end
)
while epoch < num_epochs do
	T =0
	while T < TMAX do 
      		T = T + 1
      		pool:addjob(
         	function()
            		net2:zeroGradParameters()
			vnet2:zeroGradParameters()
			local weight, gradient = net2:parameters()
			local vweight, vgradient = vnet2:parameters()
			for k, v in pairs(w) do
				weight[k]:copy(w[k])
			end
			for k, v in pairs(vw) do
				vweight[k]:copy(vw[k])
			end
			if game:isEpisodeFinished() then
				game:newEpisode()
				net2:clearState()
				vnet2:clearState()
			end	      
			j = 1
			tmax = 11  
			states = {}
			outputs = {}
			values = {}
			rewards = {}
			repeat
				state = image.scale(game:getState().screenBuffer:double(),64,64)
				action = net2:forward(state)
				value = vnet2:forward(state)
				reward = game:makeAction(torch.totable(action:int())[1])
				reward = torch.Tensor({reward})
				states[j] = state
				outputs[j] = action:int()
				values[j] = value
				rewards[j] = reward
				j = j + 1 
			until game:isEpisodeFinished() or j == tmax 
			j = j-1
			returns = values[j]
			if game:isEpisodeFinished() then
				returns = 0
				print("total reward : " .. game:getTotalReward() .." : and thread_id: ".. __threadid)
			end
			criterion = nn.MSECriterion()
			repeat
				returns = rewards[j] + 0.99 * returns
				rew = returns - values[j]
				net2:reinforce(rew)
				net2:backward(states[j])
				criterion:forward(values[j], rewards[j])
				local grad = criterion:backward(values[j], rewards[j])
				vnet2:backward(states[j], grad)
				j = j-1
			until j == 0
			return gradient, vgradient
	         end,
	         function(gradient, vgradient)		
	         	net:zeroGradParameters()
			vnet:zeroGradParameters()
			for k, v in pairs(gradient) do
				g[k]:copy(gradient[k])
			end
			for k, v in pairs(vgradient) do
				vg[k]:copy(vgradient[k])
			end
			for k, v in pairs(g) do
				rg[k]:mul(row)
				rg[k]:add(torch.mul(torch.pow(g[k],2), 1- row))
				g[k]:cdiv(torch.sqrt(torch.add(rg[k], 0.00000001)))
			end
			for k, v in pairs(vg) do
				rvg[k]:mul(row)
				rvg[k]:add(torch.mul(torch.pow(vg[k],2), 1- row))
				vg[k]:cdiv(torch.sqrt(torch.add(rvg[k], 0.00000001)))
			end		
			net:updateParameters(learning_rate)
			vnet:updateParameters(learning_rate)
		end
	      )
	      
	end
	learning_rate = learning_rate * (1 - (epoch/ num_epochs))
	epoch = epoch +1
	print (epoch)
end
if pool:hasjob() then
      pool:dojob() 
end
torch.save("net.dat", net)
torch.save("vnet.dat", vnet)
print('PASSED')
pool:terminate()


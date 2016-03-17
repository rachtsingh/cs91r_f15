nn = require "nn"
require "cutorch"
require "cunn"
require "hdf5"

cmd = torch.CmdLine()
cmd:option('-gram', 3, 'gram size')
cmd:option('-hidden', 10, 'hidden layer size')
cmd:option('-embedding', 20, 'embedding size')
-- TODO: make learning rate, file name CMD options
params = cmd:parse(arg)

-- For renormalizing the lookup tables
function renorm(data, th)
    local size = data:size(1)
    for i = 1, size do
        local norm = data[i]:norm()
        if norm > th then
            data[i]:div(norm/th)
        end
    end
end

------------------------
-- Read in data
------------------------
myFile = hdf5.open('language5.hdf5', 'r')
train = myFile:read('train'):all()
train_t = myFile:read('train_t'):all()
test = myFile:read('test'):all()
test_t = myFile:read('test_t'):all()
valid = myFile:read('valid'):all()
valid_t = myFile:read('valid_t'):all()

nV = 10000
c = params['gram']
d = params['embedding']
h = params['hidden']

batch_size = 32

------------------------
-- Build the model
------------------------

model = nn.Sequential()
E = nn.LookupTable(nV, d)
model:add(E)
model:add(nn.View(batch_size, c*d))
U = nn.Linear(c*d, h)
model:add(U)
model:add(nn.Tanh())
V = nn.Linear(h, nV)
model:add(V)
model:add(nn.LogSoftMax())
model:cuda()

criterion = nn.ClassNLLCriterion()
criterion:cuda()

function validate(inputs, outputs)
    local sum = 0
    local size = 0
    for i = 1,inputs:size(1)/batch_size do
        local input = inputs:narrow(1, (i-1)*batch_size+1, batch_size):cuda()
        local target = outputs:narrow(1, (i-1)*batch_size+1, batch_size):cuda()
        local out = model:forward(input)
        local probs = out:exp()
        for j = 1, batch_size do
            local prob = probs[j][target[j]]
            sum = sum + math.log(prob)   
        end
        size = size + input:size(1)
    end
    return math.exp((-1/inputs:size(1)) * sum)
end

------------------------
-- Train the model
------------------------
model:reset()
-- reset the bias and gradBias
V.bias:zero()
V.gradBias:zero()

learning_rate = .05
num_epochs = 50
last_perp = 0
normalization_rate = 1

for epoch = 1, num_epochs do
    nll = 0
    for j = 1, train:size(1)/batch_size do
        model:zeroGradParameters()
        input = train:narrow(1, (j-1)*batch_size+1, batch_size):cuda()
        target = train_t:narrow(1, (j-1)*batch_size+1, batch_size):cuda()
        
        out = model:forward(input)
        nll = nll + criterion:forward(out, target)

        deriv = criterion:backward(out, target)
        model:backward(input, deriv)
        V.gradBias:zero()
        model:updateParameters(learning_rate)
    end

    -- every so often normalize all the output vectors
    if epoch % normalization_rate == 0 then
        for i=1,nV do
            V.weight[i]:div(V.weight[i]:norm() + 1e7)
        end
    end

    -- Calculate the perplexity, if it has increased since last
    -- epoch, half the learning rate
    perplexity = validate(valid, valid_t)
    if last_perp ~= 0 and perplexity > last_perp then
        learning_rate = learning_rate / 2
    end
    last_perp = perplexity

    -- Renormalize the weights of the lookup table
    renorm(E.weight, 1) -- th = 1 taken from Sasha's code
    print("Epoch:", epoch, nll, perplexity)
end


----------------------
-- Compute predictions for test
----------------------

preds = torch.LongTensor(test:size(1))
for j = 1, test:size(1)/batch_size do
    input = test:narrow(1, (j-1)*batch_size+1, batch_size):cuda()
    out = model:forward(input)
    y,i = torch.max(out:float(), 2)
    for k = 1, batch_size do
        preds[(j-1)*batch_size+1+k] = i[k]
    end
end

file = hdf5.open("language125_preds.hdf5","w")
file:write("predictions", preds)
file:close()

torch.save("model125.nn", model)

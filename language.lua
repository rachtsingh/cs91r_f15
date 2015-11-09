nn = require "nn"
require "cutorch"
require "cunn"
require "hdf5"

cmd = torch.CmdLine()
cmd:option('-gram', 3, 'gram size')
cmd:option('-hidden', 10, 'hidden layer size')
cmd:option('-embedding', 20, 'embedding size')
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
myFile = hdf5.open('language.hdf5', 'r')
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
learning_rate = 1
num_epochs = 10

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
        model:updateParameters(learning_rate)
    end
    perplexity = validate(valid, valid_t)
    renorm(E.weight, 1) -- th = 1 taken from Sasha's code
    print("Epoch:", epoch, nll, perplexity)
end


----------------------
-- Compute predictions for test
----------------------

preds = torch.ones(1)
for j = 1, test:size(1)/batch_size do
    input = test:narrow(1, (j-1)*batch_size+1, batch_size):cuda()
    out = model:forward(input)
    y,i = torch.max(out, 2)
    preds = torch.cat(preds, i, 1)
end

file = hdf5.open("language_preds.hdf5","w")
file:write("predictions", preds)
file:write("E", E)
file:write("V", V)
file:write("U", U)
file:close()

print(validate(test, test_t))

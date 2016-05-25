import caffe
from caffe.proto import caffe_pb2
import tools
import yelp_multilabel_datalayers
import network
import numpy as np
import os.path as osp
import utils
import tempfile

workdir = '.'
data_root = "../data"

#def setup():
#    solverproto = tools.CaffeSolver(trainnet_prototxt_path = osp.join(workdir, "trainnet.prototxt"), testnet_prototxt_path = osp.join(workdir, "valnet.prototxt"))
#    solverproto.sp['display'] = "1"
#    solverproto.sp['base_lr'] = "0.0001"
#    solverproto.write(osp.join(workdir, 'solver.prototxt'))

def create_nets(nclasses, batch_size):
    # write train net.
    with open(osp.join(workdir, 'trainnet.prototxt'), 'w') as f:
        # provide parameters to the data layer as a python dictionary. Easy as pie!
        data_layer_params = dict(batch_size = batch_size, im_shape = [227, 227], split = 'train', data_root = data_root, n_classes = nclasses)
        train_prototxt, train_filename = network.varnet(data_layer_params, 'YelpMultilabelDataLayerSync', train=True, num_classes=nclasses)

    # write test net.
    with open(osp.join(workdir, 'testnet.prototxt'), 'w') as f:
        data_layer_params = dict(batch_size = batch_size, im_shape = [227, 227], split = 'test', data_root = data_root, train=False, n_classes=nclasses)
        test_prototxt, test_filename = network.varnet(data_layer_params, 'YelpMultilabelDataLayerSync', train=False, num_classes=nclasses)

    return train_filename, test_filename


def create_solver(train_net_path, test_net_path=None, caffe_root=".", base_lr=0.001):
    """
    solver = caffe.SGDSolver(osp.join(workdir, 'solver.prototxt'))
    solver.net.copy_from(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
    solver.test_nets[0].share_with(solver.net)
    solver.step(1)

    return solver
    """
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 1000  # Test after every 1000 training iterations.
        s.test_iter.append(100) # Test on 100 batches each time we test.

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1

    # total number of traning iterations
    # One iteration is a forward-backward pass of one minibatch
    s.max_iter = 10000     # # of times to update the net (training iterations)

    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 2000

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 100 iterations.
    s.display = 100

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 1K iterations
    s.snapshot = 1000
    s.snapshot_prefix = 'snapshots' #caffe_root + 'models/yelp_alexnet/finetune_yelp_alexnet'

    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    # Write the solver to a temporary file and return its filename.
    with open("yelp_multilabel_solver.prototxt", "w") as f:
        f.write(str(s))
        return f.name


def hamming_distance(gt, est):
    # Use Hamming Distance to find multilabel loss
    return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))


def check_accuracy(net, num_batches, batch_size):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data
        #ests = net.blobs['score'].data > 0
        ests = net.blobs['loss3/classifierx'].data > 0
        for gt, est in zip(gts, ests): #for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)


def run_solvers(niter, solvers, disp_interval=10, batch_size=128):
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""
    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            #loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy() for b in blobs)
            acc[name][it] = check_accuracy(s.test_nets[0], 20000 / batch_size, batch_size)

        if it % disp_interval == 0 or it + 1 == niter:
            #loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' % (n, loss[n][it], np.round(100*acc[n][it])) for n, _ in solvers)
            acc_disp = '; '.join('%s: acc=%2d%%' % (n, np.round(100*acc[n][it])) for n, _ in solvers)
            print 'Solver Iteration %3d) %s' % (it, acc_disp)

    # Save the learned weights from both nets.
    weight_dir = "./weights"
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = osp.join(weight_dir, filename)
        s.net.save(weights[name])
    return loss, acc, weights



def fine_tune():
    weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    data_layer_params = dict(batch_size = 128, im_shape = [227, 227], split = 'train', data_root = data_root)
    prototxt, net_filename = network.varnet(data_layer_params, 'YelpMultilabelDataLayerSync')

    untrained_yelp_net = caffe.Net(net_filename, weights, ca)

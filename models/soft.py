import numpy as np
import torch
import torch.nn as nn

class SoftInternalNode:
    def __init__(self, depth, tree, phi_numel=1):
        """
        An internal node of the soft-tree.

        Parameters
        ----------
        depth : int
            the depth of the node. "root" has depth 0 and it increases by 1.

        path_probability : float
            multiplication of probabilities (`self.prob`) of nodes that lead
            the way from the root to this node

        tree : SoftDecisionTree
            the tree that node belongs to

        phi_numel : int, optional, default=1
            Number of elements residing in each leaf node's phi. If 1, one-step
            ahead forecasting is aimed, else multi-step (direct). Passed to
            `make_children`.
        """
        self.depth = depth
        self.tree = tree

        self.is_leaf = False
        self.leaf_acc = []
        self.dense = nn.Linear(tree.input_dim, 1)
        # breakpoint()
        self.make_children(phi_numel=phi_numel)

    def make_children(self, phi_numel=1):
        """
        Every internal node is to have children and they are produced here.
        If at the penultimate depth, children will be leaves.
        """
        if self.depth + 1 == self.tree.depth:
            # We go 1 level deeper & reach the tree depth; they'll be leaves
            self.left = SoftLeafNode(self.tree, phi_numel=phi_numel)
            self.right = SoftLeafNode(self.tree, phi_numel=phi_numel)
        else:
            self.left = SoftInternalNode(self.depth + 1, self.tree,
                                         phi_numel=phi_numel)
            self.right = SoftInternalNode(self.depth + 1, self.tree,
                                          phi_numel=phi_numel)

    def forward(self, x):
        """
        Sigmoid softens the decision here.
        """
        return torch.sigmoid(self.dense(x))

    def calculate_probabilities(self, x, path_probability):
        """
        Produces the path probabilities of all nodes in the tree as well as
        the values sitting at the leaves. This is called only on the root node.
        """
        self.prob = self.forward(x)
        self.path_probability = path_probability
        left_leaf_acc = (self.left.calculate_probabilities(x,
                                                           path_probability
                                                           * (1-self.prob)))
        right_leaf_acc = (self.right.calculate_probabilities(x,
                                                             path_probability
                                                             * self.prob))
        self.leaf_acc.extend(left_leaf_acc)
        self.leaf_acc.extend(right_leaf_acc)
        return self.leaf_acc

    def reset(self):
        """
        Intermediate results i.e. leaf accumulations would cause "you're trying
        to backward the graph second time.." error so we "free" them here.
        """
        self.leaf_acc = []
        self.left.reset()
        self.right.reset()


class SoftLeafNode:
    def __init__(self, tree, phi_numel=1):
        """
        A leaf node of the soft-tree.

        Parameters
        ----------
        tree : SoftDecisionTree
            the tree that node belongs to
        phi_numel : int, optional, default=1
            Number of elements residing in each leaf node's phi. If 1, one-step
            ahead forecasting is aimed, else multi-step (direct).
        """
        self.tree = tree
        # breakpoint()
        self.phi = nn.Parameter(torch.randn(size=(phi_numel,)))
        self.is_leaf = True

    def calculate_probabilities(self, x, path_probability):
        """
        Since leaf, directly return its path_probability along with the value
        sitting at it.
        """
        return [[path_probability, self.phi]]

    def reset(self):
        """
        Keep the harmony
        """
        return


class SoftDecisionTree(nn.Module):
    def __init__(self, depth, input_dim, phi_numel=1):
        """
        A soft binary decision tree; kind of a mix of what are described at
            1) https://www.cs.cornell.edu/~oirsoy/softtree.html and
            2) https://arxiv.org/abs/1711.09784

        Parameters
        ----------
        depth : int
            the depth of the tree. e.g. 1 means 2 leaf nodes.

        input_dim : int
            number of features in the incoming input vector. (needed because
            LazyLinear layer is not available yet in this version of PyTorch,
            which is 1.7)

        phi_numel : int, optional, default=1
            Number of elements residing in each leaf node's phi. If 1, one-step
            ahead forecasting is aimed, else multi-step (direct).
        """
        super().__init__()
        self.depth = depth
        self.input_dim = input_dim
        self.phi_numel = phi_numel
        # breakpoint()
        self.root = SoftInternalNode(depth=0, tree=self, phi_numel=phi_numel)
        self.collect_trainables()

    def collect_trainables(self):
        """
        Need to say PyTorch that we need gradients calculated with respect to
        the internal nodes' dense layers and leaf nodes' values. (since nodes
        are not an nn.Module subclass).
        """
        nodes = [self.root]
        self.module_list = nn.ModuleList()
        self.param_list = nn.ParameterList()
        while nodes:
            node = nodes.pop(0)
            if node.is_leaf:
                # breakpoint()
                self.param_list.append(node.phi)
            else:
                nodes.extend([node.left, node.right])
                self.module_list.append(node.dense)

    def forward(self, xs):
        """
        y_hat  = sum(pp_l * phi_l for l in leaf_nodes of self)
        """
        leaf_acc = self.root.calculate_probabilities(xs, path_probability=1)
        pred = torch.zeros(xs.shape[0], self.phi_numel)
        for path_probability, phi in leaf_acc:
            pred += phi * path_probability
        # don't forget to clean up the intermediate results!
        self.root.reset()
        return pred


class SoftGBM(nn.Module):
    def __init__(self, num_trees, tree_depth, input_dim, shrinkage_rate,
                 phi_numel=1):
        """
        Soft gradient boosting machine i.e. a GBM where base learners are soft
        binary decision trees

        Parameters
        ----------
        num_trees : int
            number of weak learners

        tree_depths : int or list of ints
            depth of each tree. If int, repeated. Else, must be a container
            of length `num_trees`.

        input_dim : int
            number of features in the input

        shrinkage_rate : float
            ~learning rate that determines the contribution of base learners.
            (not to be mixed with what the SGD-like optimizer will use)

        phi_numel : int, optional, default=1
            Number of elements residing in each leaf node's phi. If 1, one-step
            ahead forecasting is aimed, else multi-step (direct).
        """
        super().__init__()
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.input_dim = input_dim
        self.shrinkage_rate = shrinkage_rate
        self.phi_numel = phi_numel
        # breakpoint()
        self.trees = nn.ModuleList([SoftDecisionTree(depth=tree_depth,
                                                     input_dim=input_dim,
                                                     phi_numel=phi_numel)
                                    for _ in range(num_trees)])
        self.weak_predictions = []
        self.loss_fun = nn.MSELoss()
        self.batch_losses = []

    def forward(self, x):
        self.eval()
        """
        Traverse through all the trees and ask their prediction; shrink it and
        add to
        """
        # breakpoint()
        self.weak_predictions = []
        overall_pred = torch.zeros(x.shape[0], self.phi_numel)
        for tree in self.trees:
            out = tree(x)
            overall_pred += self.shrinkage_rate * out
            self.weak_predictions.append(out)
        return overall_pred

    def get_loss(self, true):
        """
        Computes total loss i.e. sum of losses from each tree.
        """
        # the residuals are targets and it starts with y itself
        resid = true
        total_loss = torch.zeros(1)

        # for each tree..
        for j in range(self.num_trees):
            # get prediction of this tree
            pred_j = self.weak_predictions[j]
            shrunk_pred_j = self.shrinkage_rate * pred_j
            shrunk_pred_j = shrunk_pred_j.ravel()
            loss_j = self.loss_fun(shrunk_pred_j, resid)

            # update loss and residual
            total_loss += loss_j
            resid = resid - shrunk_pred_j

        return total_loss

    def train_(self, train_loader, optimizer, scheduler=None, num_epochs=100,
               print_every_e_epoch=1, print_every_b_batch=100,
               valid_loader=None, patience=None, mean=None, std=None):
        # save to use later in forecasting
        self.train_loader = train_loader

        preds = []

        # train mode on..
        self.train()
        validate = valid_loader is not None
        
        for epoch in range(num_epochs):
            for batch_idx, (xb, yb) in enumerate(train_loader):
                
                if not batch_idx % 100:
                    print(batch_idx)
                
                # FiLiP ZuBU:
                self(xb)                               # F-orward pass
                loss = self.get_loss(yb)                   # L-oss computation

                with torch.no_grad():
                    pred = self(xb)
                    
                optimizer.zero_grad()                      # Z-ero gradients
                loss.backward()                            # B-ackward pass
                optimizer.step()                           # U-pdate parameters

                preds.append(pred.detach().numpy()*std + mean)

                if validate:
                    pass
                self.batch_losses.append(loss)
                
            if scheduler is not None:
                scheduler.step()
                    
        else:
            return np.concatenate(preds, axis=0).ravel()

    def predict(self, X):
        """
        One step ahead forward
        """
        return self.forward(X)

import warnings
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from itertools import chain
from torch.utils.data import DataLoader

from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen

from utils import (_get_mixed_design_mat_part, _get_ordinal_design_mat_part,
                   _get_seasonal_design_mat_part, ordinal_diff, seasonal_diff)

from statsmodels.tsa.arima.params import SARIMAXParams
from statsmodels.tsa.arima.specification import SARIMAXSpecification

from utils import _check_endog, _check_exog, shift


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
    
    
#%% JOINT MODEL
    
class JointModel(nn.Module):
    
    def __init__(self, num_trees, tree_depth, input_dim, shrinkage_rate, mean, std,
                 endog=None, exog=None, trend="n", order=(1, 0, 0), seas_order=(0, 0, 0, 0), phi_numel=1):
        
        super().__init__()    
            
        # Base SARIMAX Parameters
        
        # Validate endog & exog
        self.endog = _check_endog(endog)
        self.exog = _check_exog(exog)
    
        # Save them to use in case of differencing
        self.the_endog = self.endog.copy()
        self.the_exog = self.exog.copy() if exog is not None else None
    
        # Orders are attributes too
        self.order = order
        self.seas_order = seas_order
    
        # "has" a specification and params (helps validate orders, also)
        self.spec = SARIMAXSpecification(self.the_endog, self.the_exog,
                                         self.order, self.seas_order)
        self.params = SARIMAXParams(self.spec)
    
        # If P == D == Q == 0, m stays the same; but should be 0, too
        if self.seas_order[:3] == (0, 0, 0):
            self.seas_order = (0, 0, 0, 0)
    
        # After validation, unpack order
        self.p, self.d, self.q = self.order
        self.P, self.D, self.Q, self.seas_period = self.seas_order
    
        # For convenience
        self.m = self.seas_period
        
        # SARIMAX Pytorch Parameters
        
        if trend not in ("n", "c", "t", "ct"):
            raise ValueError(
                f"Trend must be one of `n`, `c`, `t`, `ct`; got {trend}"
            )

        self.trend = trend
        
        self.mean = mean
        self.std = std
        
        # Standardize the endog
        self.endog = ((self.endog - self.mean) / self.std)

        # Save original mean and std
        self._orig_endog_mean = self.mean
        self._orig_endog_std = self.std

        # Revise the_endog
        self.the_endog = self.endog.copy()

        # Torch related
        self.design_matrix = self._get_design_mat()
        self.Linear = torch.nn.Linear(self.design_matrix.shape[1], 1)
        self.loss = nn.MSELoss()
        
        # Save Training
        self.endog_train = self.endog.copy()
        self.exog_train = self.exog.copy()
        self.design_matrix_train = self.design_matrix.copy()
        self.resid_train = self.resid.copy()
        
        # Others
        self.preds_in_sample = None
        self.raw_preds_in_sample = None
        
        self._set_params()
        
        # Soft GBM Parameters
        
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
        
        
    # FUNCTIONS RELATED TO THE GRADIENT BOOSTING MODEL
    
    
    def forward_tree(self, x):
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

        
    def predict_tree(self, X):
        """
        One step ahead forward
        """
        return self.forward_tree(X)
    
    
    # FUNCTIONS RELATED TO THE SARIMAX
    
    
    def undo_differences(self, preds):
        """
        Undoes the differences' effect on the predictions. Model that predicts
        over the differenced series gets back the integrated predictions.

        Parameters
        ----------
        preds : array_like
            The "raw" predictions made on stationary data.

        Returns
        -------
            The integrated predictions or `preds` itself if d = D = 0.
        """
        numel_preds = preds.size

        # Handle ordinary differences' undo
        if self.d != 0:
            preds += sum(
                        shift(
                            data=ordinal_diff(self.the_endog, i),
                            crop=True)[-numel_preds:]
                        for i in range(self.d))

        # Handle seasonal differences' undo
        if self.D != 0:
            ordi_diffed_endog = ordinal_diff(self.the_endog, self.d)
            preds += sum(
                        shift(
                            data=seasonal_diff(
                                     ordi_diffed_endog, i, self.seas_period
                                ),
                            periods=self.seas_period, crop=True
                        )[-numel_preds:]
                        for i in range(self.D)
                    )
        return preds
    
    
    def predict_in_sample(self, start=None, end=None, raw=False, std=False):
        """
        Generates in-sample i.e. training predictions.

        Parameters
        ----------
        start : int, optional  TODO: add datetime-like support
            The starting time of is-predictions. Default is the beginning.

        end : int, optional  TODO: add datetime-like support
            The end time of is-predictions. Default is the last observation.

        raw : bool, optional, default: False
            If True, then the predictions that are about the differenced data
            are returned. If d == D == 0, this has no effect. If not, then
            the integrated predictions are returned.

        Returns
        -------
            ndarray of length (end-start+1).
        """

        # Start defaults to beginning
        # don't check `end is None` as None gives the end in slicing anyway
        if start is None:
            start = 0

        # Get the "raw" predictions
        with torch.no_grad():
            preds = self.forward_sarimax(torch.Tensor(self.design_matrix)).detach().numpy().ravel()

        if self.raw_preds_in_sample is None:
            self.raw_preds_in_sample = preds.copy()
        if raw:
            return preds[start:end]

        # Undo ordinary and seasonal differences
        preds = self.undo_differences(preds)

        if std == True:
            return preds[start:end]

        # As we fitted to normalized endog, undo that
        preds = preds * self._orig_endog_std + self._orig_endog_mean

        # Store so that if asked again, give from "cache"
        self.preds_in_sample = preds

        return preds[start:end]
    
    @property
    def ar_params(self):
        empty = np.array([])
        p = self.order[0]
        if p > 0:
            params = self.Linear.weight.detach().numpy().T.ravel()
            params = params[:p]
        else:
            params = empty
        return params

    @property
    def ma_params(self):
        empty = np.array([])
        if self.q > 0:
            lower = self.p + self.P + self.p*self.P
            params = self.Linear.weight.detach().numpy().T.ravel()
            params = params[lower:lower+self.q]
        else:
            params = empty
        return params

    @property
    def seas_ar_params(self):
        empty = np.array([])
        if self.P > 0:
            lower = self.p
            params = self.Linear.weight.detach().numpy().T.ravel()
            params = params[lower:lower+self.P]
        else:
            params = empty
        return params

    @property
    def seas_ma_params(self):
        empty = np.array([])
        if self.Q > 0:
            lower = self.p + self.P + self.p * self.P + self.q
            params = self.Linear.weight.detach().numpy().T.ravel()
            params = params[lower:lower + self.Q]
        else:
            params = empty
        return params

    @property
    def drift(self):
        if self.trend not in ("t", "ct"):
            warnings.warn("No linear trend specified.")
            return np.array([])
        params = self.Linear.weight.detach().numpy().T.ravel()
        return params[-1]

    @property
    def exog_params(self):
        """
        Notes
        -----
            Doesn't include bias or trend components.
        """

        if self.exog is None:
            warnings.warn("No exogenous variable provided.")
            return np.array([])

        # Extract exogenous coeffs
        num_side_info = self.exog.shape[1]
        lower = (self.p + self.P + self.p * self.P + self.q + self.Q +
                 self.q * self.Q)

        params = self.Linear.weight.detach().numpy().T.ravel()
        return params[lower:lower + num_side_info]


    def _set_params(self):
        """
        Sets SARIMAXParams of the object.
        .exog_params *does* include bias and trend.
        """
        params = self.params
        params.ar_params = self.ar_params
        params.ma_params = self.ma_params
        params.seasonal_ar_params = self.seas_ar_params
        params.seasonal_ma_params = self.seas_ma_params
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params.exog_params = np.r_[self.drift, self.exog_params]


    def _get_design_mat(self):
        """
        Generates a design matrix of dimension N x (p+P+pP+q+Q+qQ+k+1). The
        endog-related entries are that of differenced data, and the residual-
        related entries are that of estimated ones (e.g. via HR).

        Returns
        -------
        The design matrix which has 8 parts (max): p part, P part, pP
                part, q part, Q part, qQ part, k part and trend part.
                If any of these is zero, dimension diminishes.
        """
        # Apply differences and get the number of observations after
        if self.d != 0:
            self.endog = ordinal_diff(self.endog, d=self.d)

            if self.exog is not None:
                self.exog = ordinal_diff(self.exog, d=self.d)

        if self.D != 0:
            self.endog = seasonal_diff(self.endog, D=self.D,
                                       m=self.seas_period)

            if self.exog is not None:
                self.exog = seasonal_diff(self.exog, D=self.D,
                                          m=self.seas_period)

        N = self.endog.size

        # Get estimates for residuals if we have MA terms
        if self.q > 0 or self.Q > 0:
            # via hannan-risanen
            reduced_ar_poly = list(chain.from_iterable([[1] * self.p,
                                   *[[0]*(self.seas_period - self.p - 1) +
                                   [1]*(self.p + 1)
                                   for _ in range(self.P)]]))

            reduced_ma_poly = list(chain.from_iterable([[1] * self.q,
                                   *[[0]*(self.seas_period - self.q - 1) +
                                   [1]*(self.q + 1)
                                   for _ in range(self.Q)]]))

            hr, hr_results = hannan_rissanen(self.endog,
                                             ar_order=reduced_ar_poly,
                                             ma_order=reduced_ma_poly,
                                             demean=False)
            hr_resid = hr_results.resid
            resid_numel = hr_resid.size
            self.resid = np.r_[np.zeros(N-resid_numel), hr_resid]

            # via statsmodels' sarimax
            # resid = sm.tsa.SARIMAX(
            #                        self.the_endog, self.the_exog,
            #                        order=self.order,
            #                        seasonal_order=self.seas_order
            #                       ).fit(disp=False).resid
            # self.resid = resid[self.d + self.seas_period * self.D:]

        # The 8 parts begin
        # Account for non-existent parts
        empty_part = np.array([]).reshape(N, 0)

        # AR terms
        # p part: Ordinal AR order part that accounts for \phi_i
        p_part = (
            _get_ordinal_design_mat_part(self.endog, self.p)
            if self.p > 0
            else empty_part
        )

        # P part: Seasonal AR order part that accounts for \PHI_i
        P_part = (
            _get_seasonal_design_mat_part(self.endog, self.P, self.seas_period)
            if self.P > 0
            else empty_part
        )

        # pP part: Mixed AR order part, result of multiplication
        pP_part = (
            _get_mixed_design_mat_part(
                P_part, self.p, self.P, self.seas_period
            )
            if self.p * self.P > 0
            else empty_part
        )

        # MA terms
        # q part: Ordinal MA order part that accounts for \theta_j
        q_part = (
            _get_ordinal_design_mat_part(self.resid, self.q)
            if self.q > 0
            else empty_part
        )

        # Q part: Seasonal MA order part that accounts for \THETA_j
        Q_part = (
            _get_seasonal_design_mat_part(self.resid, self.Q, self.seas_period)
            if self.Q > 0
            else empty_part
        )

        # qQ part: Mixed MA order part, result of multiplication
        qQ_part = (
            _get_mixed_design_mat_part(
                Q_part, self.q, self.Q, self.seas_period
            )
            if self.q * self.Q > 0
            else empty_part
        )

        # Exog
        exog_part = (
            self.exog
            if self.exog is not None
            else empty_part
        )

        # Deterministic liner trend
        trend_part = (
            np.arange(1, N + 1).reshape(N, 1)
            if self.trend in ("t", "ct")
            else empty_part
        )

        # Combine them all
        return np.hstack(
            (p_part, P_part, pP_part, q_part, Q_part, qQ_part,
                exog_part, trend_part)
        )
    
    def forward_sarimax(self, X):
        """
        Parameters
        ----------
        X : Design Matrix

        Returns
        -------
        Prediction

        """
        return self.Linear(X)
    
    def forecast_sarimax(self, endog, exog):
        """
        Get out-of-sample forecasts.

        Parameters
        ----------
        steps : int, optional -> defaults to 1
            The step size into future till which we forecast.
        method: str, optional
            The forecast method. "rec" for recursive (or plug-in), "direct" for
            multi-step direct, "rectify" for combination. Defaults to "rec".
        """

        endog = (endog - self.mean) / self.std

        the_endog_temp = endog
        the_exog_temp = exog

        endog = np.concatenate([self.the_endog, endog])
        exog = np.vstack((self.the_exog, exog))

        # Apply differences and get the number of observations after
        if self.d != 0:
            endog = ordinal_diff(endog, d=self.d)

            if exog is not None:
                exog = ordinal_diff(exog, d=self.d)

        if self.D != 0:
            endog = seasonal_diff(endog, D=self.D,
                                       m=self.seas_period)

            if exog is not None:
                exog = seasonal_diff(exog, D=self.D,
                                          m=self.seas_period)

        endog = endog[-len(the_endog_temp):]
        exog = exog[-len(the_exog_temp):,:]

        test_size = len(endog)

        # # Merge with Training Data        
        # self.endog = np.concatenate([self.endog, endog])
        # self.exog = np.vstack((self.exog, exog))

        # Unpack orders for ease
        p, d, q = self.order
        P, D, Q, m = self.seas_order

        # Get the reduced i.e. multiplied AR and MA parameters' estimates
        tmp_ar_coeffs = -self.params.reduced_ar_poly.coef[1:]
        tmp_ar_size = tmp_ar_coeffs.size
        ar_coeffs = np.r_[tmp_ar_coeffs, np.zeros((m*P+p)-tmp_ar_size)]

        tmp_ma_coeffs = self.params.reduced_ma_poly.coef
        tmp_ma_size = tmp_ma_coeffs.size
        ma_coeffs = np.r_[tmp_ma_coeffs, np.zeros((m*Q+q+1)-tmp_ma_size)]

        # Get last mP+p observations and mQ+q residuals (as deques)
        last_obs = deque(self.endog[-1: - (m * P + p + 1):-1],
                         maxlen=m * P + p)

        preds_in_sample = self.predict_in_sample(raw=True).ravel()
        resids = self.endog - preds_in_sample
        last_resids = deque(np.r_[0, resids[-1:(m * Q + q + 1):-1]],
                            maxlen=m * Q + q + 1)

        # Forecast loop
        forecasts = np.empty(test_size)
        
        temp = self.the_endog
        
        for h in range(test_size):
            xtended_endog = temp[-(m * D + d):].tolist()
            
            # Get the h'th "raw" forecast
            forecasts[h] = ar_coeffs.dot(last_obs) + ma_coeffs.dot(last_resids) + self.exog_params.dot(exog[h])
            temp_for = forecasts[h]

            # Undo ordinary differences
            for i in range(d):
                forecasts[h] += ordinal_diff(xtended_endog, i)[-1]

            ord_diffed_endog = ordinal_diff(xtended_endog, d)

            # Undo seasonal differences
            for j in range(D):
                forecasts[h] += seasonal_diff(
                                    ord_diffed_endog, j, m
                                )[-m]

            # Prepend predictions to `y` and fix the 0 residual from previous iteration
            last_obs.appendleft(endog[h])
            
            last_resids.popleft()
            
            last_resids.appendleft(endog[h] - temp_for)
            last_resids.appendleft(0)
            
            temp = np.append(temp, the_endog_temp[h])

        # Undo standardization
        # forecasts = forecasts * self.std + self.mean

        return forecasts
    
    def batch_creator(self, batch_size):
        """
        creates batches from design matrix, for training torch based models
        """
        
        endog = self.endog.astype(np.float32)
        design_matrix = self.design_matrix.astype(np.float32)
        
        training_dataset = []

        for i in range(len(self.endog)):
            training_dataset.append(tuple([torch.from_numpy(design_matrix[i]),torch.from_numpy(np.array(endog[i]))]))
            
        training_dataset = tuple(training_dataset)
        
        if batch_size == -1:
            batch_size = len(training_dataset)

        train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)
        return train_dataloader
    
    # Joint Trainer
    
    def train_(self, train_loader_tree, num_epochs, scheduler=None, batch_size=-1,
              print_every_e_epoch=40, print_every_b_batch=10,
              valid_loader=None, patience=None, first_endog=0, optimizer1=None, optimizer2=None):

        self.train()
        train_loader = self.batch_creator(batch_size=batch_size)

        list_sarimax = []
        list_soft = []
        
        undo_diff = [first_endog]
        do_diff = [first_endog]

        for epoch in range(num_epochs):
            
            for (batch_idx, (nonlinear, linear)) in enumerate(zip(train_loader_tree, train_loader)):               
                
                xb1 = nonlinear[0]
                xb2 = linear[0]

                yb1 = nonlinear[1]
                yb2 = linear[1]
                
                pred_sarimax = self.forward_sarimax(xb2)
                pred_tree = self.forward_tree(xb1)     # F-orward pass
                
                list_sarimax.append(pred_sarimax.ravel().detach().numpy())
                list_soft.append(pred_tree.ravel().detach().numpy())
                
                pred_sarimax_ = undo_diff[-1] + pred_sarimax.detach().numpy().ravel()
                undo_diff.append(yb1.numpy())
                
                pred_tree_ = pred_tree.detach().numpy().ravel() - do_diff[-1]
                do_diff.append(pred_tree.detach().numpy().ravel())
                
                yb1 -= pred_sarimax_
                yb2 -= pred_tree_
                
                optimizer1.zero_grad()
                loss = self.loss(pred_sarimax.ravel(), yb2)
                loss.backward()
                optimizer1.step()
                
                optimizer2.zero_grad()                      # Z-ero gradients
                loss_tree = self.get_loss(yb1)        # L-oss computation
                loss_tree.backward()                  # B-ackward pass
                optimizer2.step()                           # U-pdate parameters
                
            if scheduler is not None:
                scheduler.step()
     
            self._set_params()

        return self.undo_differences(np.concatenate(list_sarimax, axis=0).ravel()) , np.concatenate(list_soft, axis=0)

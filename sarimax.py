import warnings
from collections import deque
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from statsmodels.tsa.arima.estimators.hannan_rissanen import hannan_rissanen

from utils import (_get_mixed_design_mat_part, _get_ordinal_design_mat_part,
                   _get_seasonal_design_mat_part, ordinal_diff, seasonal_diff)

from statsmodels.tsa.arima.params import SARIMAXParams
from statsmodels.tsa.arima.specification import SARIMAXSpecification

from torch.utils.data import DataLoader
from utils import _check_endog, _check_exog, shift


class SARIMAX_SGD(nn.Module):
    """
    Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors
    model. Solved with Stochastic Gradient Descent.
    """

    def __init__(self, endog, exog=None, mean=0, std=1, trend="n", order=(1, 0, 0),
                 seas_order=(0, 0, 0, 0)):
        """
        Parameters
        ----------
        endog: array_like
            The time series array.

        exog: array_like, optional -> defaults to None
            The exogenous regressors i.e. side information. Shape is n_obs x k.

        trend: string, optional -> defaults to "n" meaning no trend component
            possible settings are "ct", "c", "n", "t"

        order: sequence of three ints, optional -> defaults to (1, 0, 0)
            The (p, d, q) of non-seasonal ARIMA part's order. All items must be
            nonnegative integers. Default is AR(1) process.

        seas_order: sequence of four ints, optional -> defaults to (0, 0, 0, 0)
            The (P, D, Q, m) of seasonal ARIMA part. All items must be
            nonnegative integers.
        """
        super().__init__()
        
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

        if trend not in ("n", "c", "t", "ct"):
            raise ValueError(
                f"Trend must be one of `n`, `c`, `t`, `ct`; got {trend}"
            )
        self.trend = trend
        
        self.mean = mean
        self.std = std
        
        # Standardize the endog
        self.endog = (self.endog - mean) / std

        # Save original mean and std
        self._orig_endog_mean = mean
        self._orig_endog_std = std

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
        # self.resid_train = self.resid.copy()
        
        # Others
        self.preds_in_sample = None
        self.raw_preds_in_sample = None
        
        self._set_params()
        
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
    
    def predict_in_sample(self, start=None, end=None, raw=False):
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
            preds = self.forward(torch.Tensor(self.design_matrix)).detach().numpy().ravel()

        if self.raw_preds_in_sample is None:
            self.raw_preds_in_sample = preds.copy()
        if raw:
            return preds[start:end]

        # Undo ordinary and seasonal differences
        preds = self.undo_differences(preds)

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
    
    
    def forward(self, X):
        
        self.eval()
        """
        Parameters
        ----------
        X : Design Matrix

        Returns
        -------
        Prediction

        """
        return self.Linear(X)
    
    
    def train_(self, num_epochs, optimizer, scheduler=None, batch_size=-1,
              print_every_e_epoch=100, print_every_b_batch=10,
              valid_loader=None, patience=None):

        self.train()

        preds = []

        for epoch in range(num_epochs):
            
            train_loader = self.batch_creator(batch_size=batch_size)
            for batch_idx, (xb, yb) in enumerate(train_loader):              

                pred = self.forward(xb)
                loss = self.loss(pred.ravel(), yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                preds.append(pred.ravel().detach().numpy())
                
            if scheduler is not None:
                scheduler.step()
                
            self._set_params()
                
        return self.undo_differences(np.concatenate(preds, axis=0).ravel())*self._orig_endog_std + self._orig_endog_mean
    
    
    def forecast(self, endog, exog):
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
        forecasts = forecasts * self.std + self.mean

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

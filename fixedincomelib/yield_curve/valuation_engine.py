"""Analytic valuation of a signed fixed-accrual cashflow on an IFR model."""

import numpy as np

from fixedincomelib.model import ModelType
from fixedincomelib.product import ProductFixedAccruedCashflow
from fixedincomelib.valuation import (
    AnalyticValParam, FundingIndexParameter, ValuationEngineProduct,
    ValuationEngineProductRegistry,
)


class ValuationEngineProductFixedAccrued(ValuationEngineProduct):
    def __init__(self, model, valuation_parameters_collection, product, request):
        super().__init__(model, valuation_parameters_collection, product, request)
        funding_parameter = valuation_parameters_collection.get_vp_from_build_method_collection(
            FundingIndexParameter._vp_type)
        self.funding_index_ = funding_parameter.get_funding_index(product.currency)
        self.cash_amount_ = product.notional * product.accrued

    def calculate_value(self) -> None:
        payment_date = self.product_.payment_date
        if payment_date < self.value_date:
            self.value_, self.cash_ = 0.0, 0.0
        elif payment_date == self.value_date:
            self.value_, self.cash_ = self.cash_amount_, self.cash_amount_
        else:
            df = self.model.discount_factor(
                self.funding_index_, payment_date, self.product_.currency.value_str)
            self.value_, self.cash_ = self.cash_amount_ * df, 0.0

    def calculate_risk(
        self, gradient: list[np.ndarray], scaler: float = 1.0, accumulate: bool = False,
    ) -> None:
        payment_date = self.product_.payment_date
        if payment_date > self.value_date:
            self.model.discount_factor_gradient_wrt_state(
                self.funding_index_, payment_date, gradient,
                scaler=scaler * self.cash_amount_, accumulate=accumulate)
            return
        # Past or same-day payments have zero risk.
        self.model.resize_gradient(gradient)
        if not accumulate:
            for block in gradient:
                block[:] = 0.0


ValuationEngineProductRegistry().register(
    (ModelType.YIELD_CURVE.to_string(), ProductFixedAccruedCashflow._product_type, AnalyticValParam._vp_type),
    ValuationEngineProductFixedAccrued,
)

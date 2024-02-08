
import QuantLib as quantl


def execute_bond_analysis():

    bond_calculation_date = quantl.Date(1,9,2023)
    quantl.Settings.instance().evaluationDate = bond_calculation_date


    bond_redemption = 110.00
    bond_face_amount = 110.0
    bond_spot_price = 49.04
    bond_conversion_price = 27.0
    bond_conversion_ratio = 3.74615 

    bond_issue_date = quantl.Date(16,4,2023)        
    bond_maturity_date = quantl.Date(16,4,2024)

    bond_settlement_days = 2
    bond_calendar = quantl.UnitedStates(quantl.UnitedStates.GovernmentBond)
    bond_coupon = 0.0575
    bond_frequency = quantl.Semiannual
    bond_tenor = quantl.Period(bond_frequency)

    bond_day_count = quantl.Thirty360(quantl.Thirty360.BondBasis)
    bond_accrual_convention = quantl.Unadjusted
    bond_payment_convention = quantl.Unadjusted

    bond_call_dates = [quantl.Date(20,8,2023)]
    bond_call_price = 100.0
    bond_put_dates = [quantl.Date(20,8,2023), quantl.Date(15,7,2023), quantl.Date(15,6,2023)]
    bond_put_price = 100.0

    bond_dividend_yield = 0.03
    bond_credit_spread_rate = 0.04 
    bond_risk_free_rate = 0.05
    bond_volatility = 0.35

    bond_callability_schedule = quantl.CallabilitySchedule()


    for bond_call_date in bond_call_dates:
       bond_callability_price  = quantl.BondPrice(bond_call_price, 
                                                quantl.BondPrice.Clean)
       bond_callability_schedule.append(quantl.Callability(bond_callability_price, 
                                           quantl.Callability.Call,
                                           bond_call_date)
                            )
    
    for bond_put_date in bond_put_dates:
        bond_puttability_price = quantl.BondPrice(bond_put_price, 
                                                quantl.BondPrice.Clean)
        bond_callability_schedule.append(quantl.Callability(bond_puttability_price,
                                                   quantl.Callability.Put,
                                                   bond_put_date))


    bond_dividend_schedule = quantl.DividendSchedule()
    bond_dividend_amount = bond_dividend_yield*bond_spot_price
    bond_next_dividend_date = quantl.Date(1,12,2004)
    bond_dividend_amount = bond_spot_price*bond_dividend_yield
    for i in range(4):
        date = bond_calendar.advance(bond_next_dividend_date, 1, quantl.Years)
        bond_dividend_schedule.append(
            quantl.FixedDividend(bond_dividend_amount, date)
        )
    bond_schedule = quantl.Schedule(bond_issue_date, bond_maturity_date, bond_tenor,
                           bond_calendar, bond_accrual_convention, bond_accrual_convention,
                           quantl.DateGeneration.Backward, False)

    bond_credit_spread_handle = quantl.QuoteHandle(quantl.SimpleQuote(bond_credit_spread_rate))
    bond_exercise = quantl.AmericanExercise(bond_calculation_date, bond_maturity_date)

    convertible_bond = quantl.ConvertibleFixedCouponBond(bond_exercise,
                                                     bond_conversion_ratio,
                                                     bond_callability_schedule, 
                                                     bond_issue_date,
                                                     bond_settlement_days,
                                                     [bond_coupon],
                                                     bond_day_count,
                                                     bond_schedule,
                                                     bond_redemption)
    bond_spot_price_handle = quantl.QuoteHandle(quantl.SimpleQuote(bond_spot_price))
    bond_yield_ts_handle = quantl.YieldTermStructureHandle(
        quantl.FlatForward(bond_calculation_date, bond_risk_free_rate, bond_day_count)
    )
    bond_dividend_ts_handle = quantl.YieldTermStructureHandle(
        quantl.FlatForward(bond_calculation_date, bond_dividend_yield, bond_day_count)
    )
    bond_volatility_ts_handle = quantl.BlackVolTermStructureHandle(
        quantl.BlackConstantVol(bond_calculation_date, bond_calendar,bond_volatility, bond_day_count)
    )

    bsm_process = quantl.BlackScholesMertonProcess(bond_spot_price_handle, 
                                               bond_dividend_ts_handle,
                                               bond_yield_ts_handle,
                                               bond_volatility_ts_handle)


    bond_credit_spread_handle = quantl.QuoteHandle(quantl.SimpleQuote(bond_credit_spread_rate))
    time_steps = 1000
    bond_engine = quantl.BinomialConvertibleEngine(bsm_process, "crr", time_steps,bond_credit_spread_handle, bond_dividend_schedule)



    convertible_bond.setPricingEngine(bond_engine)
    print("Net Present Value of the bond ", convertible_bond.NPV())

if __name__ == "__main__":
  
  execute_bond_analysis()
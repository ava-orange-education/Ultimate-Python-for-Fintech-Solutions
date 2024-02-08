import statsmodels.api as statsm
import pandas as pand

def fitOLSRegression(xarr,yarr):

	xarr = statsm.add_constant(xarr)

	fit_equation = statsm.OLS(yarr, xarr).fit()

	print(fit_equation.summary())


if __name__ == "__main__":

   dataset = pand.read_csv('dataset1.csv')

   xarr = dataset['x'].tolist()
   yarr = dataset['y'].tolist()

   fitOLSRegression(xarr,yarr)

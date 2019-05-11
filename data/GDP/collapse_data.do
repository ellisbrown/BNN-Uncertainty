set more off
import excel "/Users/Ethan/Google Drive/School/Fed/EECS 6699/Project/BNN-Uncertainty/data/GDP/2017-01-27.xls", sheet("data") firstrow clear

drop JTSJOL DGORDER RSAFS PPIFIS BOPTEXP BOPTIMP WHLSLRIMSA TTLCONS BUSINV GACDISA066~Y PCEC96

foreach var in PAYEMS CPIAUCSL HSN1F UNRATE HOUST INDPRO DSPIC96 IR CPILFESL PCEPILFE PCEPI PERMIT TCU IQ GACDFSA066MSFRBPHI GDPC1 ULCNFB A261RX1Q020SBEA {
	gen L`var'=`var'[_n-1]
	gen L2`var'=`var'[_n-2]
	gen L3`var'=`var'[_n-3]
	gen L4`var'=`var'[_n-4]
}

drop if _n == 385
gen year = year(Date)
gen qtr = quarter(Date)
gen yq = year + qtr/4

drop Date year qtr

collapse (mean) PAYEMS-L4A261RX1Q020SBEA, by(yq)

drop if _n < 3

export delimited using "/Users/Ethan/Google Drive/School/Fed/EECS 6699/Project/BNN-Uncertainty/data/GDP/gdp_data.csv", replace


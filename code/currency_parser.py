import pandas as pd
base_dir = '../currency_rate_history/'

ars_to_usd = []
with open(base_dir + 'ars_usd_avg.txt') as file:
    for line in file:
        ars_to_usd.append(float(line[4:12]))
        
euro_usd_avg = []
with open(base_dir + 'euro_usd_avg.txt') as file:
    for line in file:
        euro_usd_avg.append(float(line[4:12]))
        
cop_usd_avr = []
with open(base_dir + 'cop_usd_avr.txt') as file:
    for line in file:
        cop_usd_avr.append(float(line[4:12]))
        
        
dkk_usd_avg = []
with open(base_dir + 'dkk_usd_avg.txt') as file:
    for line in file:
        dkk_usd_avg.append(float(line[4:12]))
        
        
gbp_usd_avg = []
with open(base_dir + 'gbp_usd_avg.txt') as file:
    for line in file:
        gbp_usd_avg.append(float(line[4:12]))
        
        
ars_to_usd = pd.DataFrame(ars_to_usd, columns=['ARS'])
euro_usd_avg = pd.DataFrame(euro_usd_avg, columns=['EUR'])
cop_usd_avr = pd.DataFrame(cop_usd_avr, columns=['COP'])
dkk_usd_avg = pd.DataFrame(dkk_usd_avg, columns=['DKK'])
gbp_usd_avg = pd.DataFrame(gbp_usd_avg, columns=['GBP'])

currency_rates = pd.concat([ars_to_usd,
                            euro_usd_avg,
                            cop_usd_avr,
                            dkk_usd_avg,
                            gbp_usd_avg,
                            ], axis=1)
    
currency_rates.to_excel(base_dir + 'currency_rates.xlsx')

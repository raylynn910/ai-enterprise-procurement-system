import sys
sys.path.append('backend')
from main import predict_supplier_risk, SupplierRiskRequest

req = SupplierRiskRequest(supplier_id='SUP-010', country='US', category='IT', lead_time_days=10)
res = predict_supplier_risk(req)
print("Result:")
print(res)

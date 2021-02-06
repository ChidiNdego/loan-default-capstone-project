import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://d4aa7520-9f53-427a-9f96-7d7ccb145c0e.southcentralus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = 'dzQpVmOR79g0kDN3K2As1sgjqTRIn5iq'

# Four sets of data to score, so we get four results back
data = {"data": [{"Column1": 140887, "firstPaymentDefault": 0, "firstPaymentRatio": 0.0, "max_amount_taken": 0, "max_tenor_taken": 1, "loanAmount": 45000, "interestRate": 5.0, "clientIncome": 125000.0, "clientAge": 41, "clientNumberPhoneCOntacts": 3013.0, "clientAvgCallsPerDay": 18.680497925311204, "loanNumber": 14, "clientGender_MALE": 1, "clientMaritalStatus_Married": 1, "clientMaritalStatus_Separated": 0, "clientMaritalStatus_Single": 0, "clientMaritalStatus_Widowed": 0, "clientLoanPurpose_education": 0, "clientLoanPurpose_house": 1, "clientLoanPurpose_medical": 0, "clientLoanPurpose_other": 0, "clientResidentialStauts_Family Owned": 0, "clientResidentialStauts_Own Residence": 0, "clientResidentialStauts_Rented": 1, "clientResidentialStauts_Temp Residence": 0, "incomeVerified_True": 1}, {"Column1": 157934, "firstPaymentDefault": 1, "firstPaymentRatio": 0.0, "max_amount_taken": 1, "max_tenor_taken": 1, "loanAmount": 34500, "interestRate": 12.5, "clientIncome": 17500.0, "clientAge": 29, "clientNumberPhoneCOntacts": 358.0, "clientAvgCallsPerDay": 3.167883211678832, "loanNumber": 3, "clientGender_MALE": 0, "clientMaritalStatus_Married": 0, "clientMaritalStatus_Separated": 0, "clientMaritalStatus_Single": 1, "clientMaritalStatus_Widowed": 0, "clientLoanPurpose_education": 0, "clientLoanPurpose_house": 0, "clientLoanPurpose_medical": 0, "clientLoanPurpose_other": 0, "clientResidentialStauts_Family Owned": 0, "clientResidentialStauts_Own Residence": 0, "clientResidentialStauts_Rented": 1, "clientResidentialStauts_Temp Residence": 0, "incomeVerified_True": 0}, {"Column1": 23515, "firstPaymentDefault": 1, "firstPaymentRatio": 0.0, "max_amount_taken": 0, "max_tenor_taken": 1, "loanAmount": 19500, "interestRate": 12.5, "clientIncome": 17500.0, "clientAge": 21, "clientNumberPhoneCOntacts": 1067.0, "clientAvgCallsPerDay": 134.88235294117646, "loanNumber": 3, "clientGender_MALE": 0, "clientMaritalStatus_Married": 0, "clientMaritalStatus_Separated": 0, "clientMaritalStatus_Single": 1, "clientMaritalStatus_Widowed": 0, "clientLoanPurpose_education": 0, "clientLoanPurpose_house": 0, "clientLoanPurpose_medical": 0, "clientLoanPurpose_other": 0, "clientResidentialStauts_Family Owned": 0, "clientResidentialStauts_Own Residence": 0, "clientResidentialStauts_Rented": 1, "clientResidentialStauts_Temp Residence": 0, "incomeVerified_True": 0}, {"Column1": 5117, "firstPaymentDefault": 1, "firstPaymentRatio": 0.7722358286335252, "max_amount_taken": 1, "max_tenor_taken": 1, "loanAmount": 200000, "interestRate": 5.0, "clientIncome": 300000.0, "clientAge": 28, "clientNumberPhoneCOntacts": 2372.0, "clientAvgCallsPerDay": 189.4017094017094, "loanNumber": 5, "clientGender_MALE": 1, "clientMaritalStatus_Married": 1, "clientMaritalStatus_Separated": 0, "clientMaritalStatus_Single": 0, "clientMaritalStatus_Widowed": 0, "clientLoanPurpose_education": 0, "clientLoanPurpose_house": 0, "clientLoanPurpose_medical": 0, "clientLoanPurpose_other": 0, "clientResidentialStauts_Family Owned": 1, "clientResidentialStauts_Own Residence": 0, "clientResidentialStauts_Rented": 0, "clientResidentialStauts_Temp Residence": 0, "incomeVerified_True": 1}]}

# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())

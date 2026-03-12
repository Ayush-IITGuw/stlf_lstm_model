from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scale_data(type = "standard"):
	"Scale the data either max-min or standard scaling"
	
	if type == "minmax":
	        return MinMaxScaler()
        elif type == "standard":
    		return StandardScaler()


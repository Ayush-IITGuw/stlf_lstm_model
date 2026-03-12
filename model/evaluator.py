from .metrics import model_report
class Evaluator:
	def evaluate(self, y_true, y_pred):
	        return model_report(y_true, y_pred)


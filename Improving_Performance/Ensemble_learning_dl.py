from collections import Counter
class ensemble_model:
  def __init__(self):

    '''
    This class takes in trained models and combines them for ensemble learning.
    The models can be added to the models list in the constructor of this class.
    it has 2 functions as of now : predict(self,test_data) and evaluate(self,test_data).
    '''

    self.models=[tf.keras.load_model('eff_net_101food.h5'),tf.keras.load_model('eff_net_b7.h5')]
    self.no_models=len(self.models)


  def predict(self,test_data):

    '''
    uses trained models to predict labels and based on max voting outputs predicted label.
    takes in test_data as input and outputs prediction labels as list.
    '''

    y_pred={}
    for i in range(self.no_models):
      y_pred[i]=np.armax(self.models[i].predict(test_data),axis=1)

    y_preds=[]
    for i in range(y_pred[0]):
      y_pred_i=[]
      for j in range(self.no_models):
        y_pred_i[i]=y_pred[j][i]
      y_pred_i=Counter(y_pred_i).most_common()[0][0]
      y_preds+=y_pred_i
    return y_preds

    def evaluate(self,test_data):

      '''
      evaluates the prediction made by the model with respect to the test labels.
      Takes in Test data and returns accuracy.
      '''

      y_preds=self.predict(test_data)
      tp=0
      for i in range(len(test_data)):
        if y_preds[i]==test_data[i]:
          tp+=1
    return tp/len(test_data)

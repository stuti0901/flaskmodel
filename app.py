from flask import Flask, render_template, request,jsonify
import flask,traceback
import os
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__,template_folder='templates')

with (open("C:\\Users\\hp\\Desktop\\flaskmodel\\MLmodel\\model.sav", "rb")) as f:
    regressor = pickle.load(f)

with open("C:\\Users\\hp\\Desktop\\flaskmodel\\MLmodel\\model_columns.pkl","rb") as f:
     model_columns = pickle.load(f)


@app.route('/', methods=['POST','GET'])
@app.route('/predict', methods=['POST','GET'])
def predict():
   
    if flask.request.method == 'POST':
       try:
           json_ = request.form.to_dict()
           print(json_)
           query_ = pd.get_dummies(pd.DataFrame(json_, index = [0]), prefix=['job_state','Sector','job_sim'], columns=['job_state','Sector','job_sim'])
           print(query_)
           query = query_.reindex(columns = model_columns, fill_value= 0)
           print(query)
           prediction = list(regressor.predict(query))
           final_val = round(prediction[0],2)
 
           #return jsonify({
           #    "prediction":str(prediction)
           #})
           return render_template('home.html', prediction = final_val)
 
       except:
           return jsonify({
               "trace": traceback.format_exc()
               })
    else:
        return render_template('home.html')

      
 
if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD']=True
    app.run(debug=True,use_reloader=True)





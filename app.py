import optuna
import pandas as pd
import streamlit as st
from joblib import load

X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

st.set_page_config(page_title="Car Price Forecast", page_icon="images/icon.png")


st.sidebar.markdown(
    """<p style="text-align: center;">
	<img src="https://i.ibb.co/YNfphXd/u-https-images-vexels-com-media-users-3-127711-isolated-preview-384e0b3361d99d9c370b4037115324b9-fla.png" alt="u-https-images-vexels-com-media-users-3-127711-isolated-preview-384e0b3361d99d9c370b4037115324b9-fla" alt="Cars" width="150" height="150">
	</p>
	""",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """<p style="font-family:Cambria; font-size: 25px; text-align: center">
	Car Price Forecast
	<hr>
	</p>""",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    label="",
    options=[
        "Introduction",
        "Overview and Pipeline",
        "Optuna Auto-Tuning",
        "Prediction Model",
    ],
)

st.sidebar.markdown("<hr>", unsafe_allow_html=True)

st.sidebar.markdown(
    """<p style="text-align: center;">
	<a href="https://github.com/joao-vitor-souza/Car-Price-Prediction">
	<img src="https://i.ibb.co/hMhgbpG/imagem-menu.png" alt="Github" width="200" height="75">
	</a>
	</p>
	""",
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """<p style="text-align: center;">
	Made with <a style="text-decoration: none" href='http://streamlit.io/'>Streamlit<a>
	</p>
	""",
    unsafe_allow_html=True,
)

if page == "Introduction":
    "---"

    st.header("Business Problem")

    text_1 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	This application aims to provide a regression model capable of estimating a vehicle's price. Suppose a used car store is interested in 
 	buying a vehicle and needs to know its real market value. Instead of using intuition and a limited knowledge of all variables that influence
  	the price, it can use this model to predict the value at which the car is commonly sold, and then, guided by this value, it can offer a lower
   	bid, aiming greater profits on resale.
	</p>"""

    text_2 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	The project was developed in a competition environment on 
	<a style="text-decoration: none" href="https://www.kaggle.com/datasets/kukuroo3/used-car-price-dataset-competition-format">Kaggle</a>, 
	so a bunch of regression models was tested, including
	<a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">Linear Regression</a>, 
	<a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html?highlight=decision%20tree%20regressor#sklearn.tree.DecisionTreeRegressor">Decision Tree Regressor</a>, 
	<a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/ensemble.html">ensemble algorithms</a> like
	<a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html?highlight=extratreesregressor#sklearn.ensemble.ExtraTreesRegressor">Extra Trees Regressor</a>, 
	<a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=random%20forest#sklearn.ensemble.RandomForestRegressor">Random Forest Regressor</a>, 
	<a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html?highlight=votingregressor#sklearn.ensemble.VotingRegressor">Voting Regressor</a>, 
	and the powerful <a style="text-decoration: none" href="https://xgboost.readthedocs.io/en/latest/python/python_api.html">XGBRegressor</a>.
	The performance metrics used were the average of the 
	<a style="text-decoration: none" href="https://en.wikipedia.org/wiki/Coefficient_of_determination">Coefficient of Determination (R²)</a>, 
	<a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html?highlight=rmse">Root Mean Squared Error (RMSE)</a>,
	and <a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html?highlight=mean%20absolute#sklearn.metrics.mean_absolute_error">Mean Absolute Error (MAE)</a>. 
	<br><br>The best model deployment was implemented in this application, but if you want to see the tuning and scores of all models you can click on this
	<a style="text-decoration: none" href=https://www.kaggle.com/code/joaovitorsilva/optuna-xgbregressor-r-0-9567?scriptVersionId=91324274">link<a>.
	</p>"""

    st.markdown(text_1, unsafe_allow_html=True)

    # No copyright image.
    st.image("images/image.jpg")
    st.info(
        'To skip directly to the model, click on "Prediction Model" option on the side bar.'
    )
    st.markdown(text_2, unsafe_allow_html=True)
    "---"

if page == "Overview and Pipeline":
    "---"

    st.header("Overview")

    st.subheader("Load")

    text_1 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	The data available on Kaggle are already splitted into train and test sets. Let's load the train sets to <i>X_train</i> and <i>y_train</i> 
 	variables using
	<a style="text-decoration: none" href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html">pandas.read_csv</a>
	and see their first five instances:
	</p>"""

    st.markdown(text_1, unsafe_allow_html=True)
    st.warning(
        "The data shown here were already pre-processed, access Kaggle to see what was done."
    )

    with st.echo():
        st.dataframe(X_train.head())
        st.dataframe(y_train.head())

    st.subheader("Description")
    with st.echo():
        st.dataframe(X_train.describe())

    text_2 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	The 25% of vehicles that consume the least are mostly hybrid vehicles, and about 50% (median) of vehicles in the database are popular 2.0 
 	vehicles with less than 5 years in 2022.
	</p>"""

    st.markdown(text_2, unsafe_allow_html=True)

    st.subheader("Correlations")

    with st.echo():
        st.dataframe(pd.concat([X_train, y_train], axis=1).corr())

    text_3 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	As we might expect, the price is positively correlated with the year of the car and the size of its engine. Moreover, it's negatively 
 	correlated with car mileage.
	</p>"""

    st.markdown(text_3, unsafe_allow_html=True)

    "---"
    st.header("Pipeline")

    text_4 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	The pipeline will
	<a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html?highlight=standardscaler#sklearn.preprocessing.StandardScaler">standardize</a> and
	<a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html?highlight=onehotencode#sklearn.preprocessing.OneHotEncoder">one hot-encode</a> 
	the features using a
	<a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html?highlight=columntransformer#sklearn.compose.ColumnTransformer">ColumnTransformer</a>:
	</p>"""

    st.markdown(text_4, unsafe_allow_html=True)

    st.code(
        """pipeline = ColumnTransformer([
	('num', StandardScaler(), ['year', 'mileage', 'engineSize']), 
        ('cat', OneHotEncoder(), ['brand', 'model', 'transmission', 'fuelType'])
        ])"""
    )

    "---"

if page == "Optuna Auto-Tuning":
    "---"

    st.header("Optuna Auto-Tuning")

    text_1 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	After training all models and evaluating them with
	<a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html?highlight=cross_val_score#sklearn.model_selection.cross_val_score">cross-validation</a> 
	we found out that the best performing regressor was the XGBRegressor. Now we will improve it using the auto-tuning framework
	<a style="text-decoration: none" href="https://optuna.org/">Optuna</a>, for this we'll define a objective function:
	</p>"""

    st.markdown(text_1, unsafe_allow_html=True)

    st.code(
        """def objective(trial):
    
    booster = trial.suggest_categorical('booster', ['gbtree', 'gblinear'])
    n_estimators = trial.suggest_int('n_estimators', 3, 500)
    learning_rate = trial.suggest_float('learning_rate', 10e-5, 1)

    xgb_reg = XGBRegressor(objective='reg:squarederror', 
                           n_estimators=n_estimators,
                           booster=booster,
                           learning_rate=learning_rate,
                           random_state=0)
    
    scores = cross_val_score(xgb_reg, X_train, y_train)
    
    r_2 = scores.mean()
        
    return r_2"""
    )

    text_2 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
    The above function will be our optimization base. It takes a parameter with an object
	<a style="text-decoration: none" href="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial">trial</a> 
	which tells how the sampling of the hyperparameters should be done. Once the hyperparameters are chosen, they are used to create an 
 	XGBRegressor object with a <i>reg:squarederror</i> regression objective, and by cross-validation we calculate and return the average R² 
  	Coefficient of Determination. The metric that is returned, in this case, the R², is the value that will be used as a reference for the 
   	optimization. The question is, what is the advantage of using a trial object? And how do we create one? As said, a trial samples the 
    hyperparameters within the spaces defined for them (categorical, float, int...), and this sampling depends on a 
	<a style="text-decoration: none" href="https://optuna.readthedocs.io/en/stable/reference/samplers.html">sampler</a>.
	The default Optuna sampler is called 
	<a style="text-decoration: none" href="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html">TPESampler</a>
	(<i>Tree-structured Parzen Estimator</i>). This sampler is better than a random sampler like the one used in the
	<a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html">RandomizedSearchCV</a> Scikit-Learn class, 
	because it applies a
	<a style="text-decoration: none" href="https://brilliant.org/wiki/gaussian-mixture-model/">Gaussian Mixture Model</a> which tends to segment
 	the sampling space into regions that are already performing well. For example, suppose the TPE chooses two values, α and β for any 
  	hyperparameter, if R² is greater for α than for β, then the sampler adjusts the parameters (means and covariances) of the Gaussian mixtures
   	in order to increase the probability of choose another value close to α in the next sampling. This is the same process used in clustering
    algorithms like
	<a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html">GaussianMixture</a>.
	That said, we will instantiate a TPE sampler with a fixed seed to allow reproducibility:
	</p>"""

    st.markdown(text_2, unsafe_allow_html=True)

    st.code("sampler = optuna.samplers.TPESampler(seed=0)")

    text_3 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	The second question has not yet been answered, how do we create a trial? We can manually create one by instantiation, but that's not 
 	practical. With this in mind, the  	
	<a style="text-decoration: none" href="https://optuna.readthedocs.io/en/stable/reference/study.html">study</a> class was developed. The 
 	objects of this class automate the passing of parameters to the objective function, not only that, it is through them that we choose the 
  	sampler and the direction of optimization of the trials. If our objective function returns a metric that needs to be decreased as an
   	<a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error">MSE</a>,
   	so we choose the direction that minimizes this error. Let's instantiate a study with the
   	<a style="text-decoration: none" href="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html#optuna.study.create_study">create_study</a> method
   	in the direction of increasing R² with a TPE sampler, moreover, aiming to export the study with
   	<a style="text-decoration: none" href="https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html">joblib.dump</a>,
   	we'll give it a name:
    </p>"""

    st.markdown(text_3, unsafe_allow_html=True)

    st.code(
        "study = optuna.create_study(study_name='study', sampler=sampler, direction='maximize')"
    )

    text_4 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	To start the optimization process, we call the 
   	<a style="text-decoration: none" href="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize">optimize</a> method
   	with the objective function and the number of trials. Instead of using the number of trials, we could code the amount of time we want the
    study to take place, but this is not reproducible, as it depends on the hardware of each machine that calls the method. Also, to avoid
    excess information on the screen, we will change the verbosity of the study with
   	<a style="text-decoration: none" href="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.logging.set_verbosity.html#optuna.logging.set_verbosity">set_verbosity</a>.
	Initially, we'll pass 30 trials and will check if the model converged to the best hyperparameters:
    </p>"""

    st.markdown(text_4, unsafe_allow_html=True)

    st.code(
        """optuna.logging.set_verbosity(optuna.logging.ERROR)
study.optimize(objective, n_trials=30)"""
    )

    text_5 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	The best hyperparameters and training scores can be accessed with the attributes:
    </p>"""

    st.markdown(text_5, unsafe_allow_html=True)

    study = load("models/study.pkl")

    with st.echo():
        st.write(study.best_params)
        st.write(study.best_value)

    text_6 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	To check if the model converged, we'll analyze the R² vs number of trials chart:
    </p>"""

    st.markdown(text_6, unsafe_allow_html=True)

    with st.echo():
        # Checking if plotly visualization is available.
        st.write(optuna.visualization.is_available())

    with st.echo():
        fig = optuna.visualization.plot_optimization_history(study)
        fig.update_layout(
            yaxis_visible=False, yaxis_showticklabels=False, font=dict(size=15)
        )

        st.plotly_chart(fig, use_container_width=True)

    text_7 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	As we can see, the model converged in less than 30 trials. Now let's see how the TPE sampler made it easier to choose the number of
 	estimators and the learning rate:
	</p>"""

    st.markdown(text_7, unsafe_allow_html=True)

    with st.echo():
        fig = optuna.visualization.plot_contour(
            study, ["n_estimators", "learning_rate"]
        )
        fig.update_layout(font=dict(size=15))

        st.plotly_chart(fig, use_container_width=True)

    text_8 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
    It is interesting to note that, even in a few attempts, the sampler easily and precisely defines the contours of high and low probability
    regions, choosing hyperparameters that generate the best possible models. Finally, let's see the importance of each hyperparameter when
    building the model:
    </p>"""

    st.markdown(text_8, unsafe_allow_html=True)

    with st.echo():
        fig = optuna.visualization.plot_param_importances(study)
        fig.update_layout(font=dict(size=15))

        st.plotly_chart(fig, use_container_width=True)

    text_9 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
	The hyperparameter that matters most for the XGBRegressor in this study is the booster. This makes sense since it basically changes the way
 	the entire model will be built. A gblinear booster uses a combination of linear functions weighted to calculate the residuals and generate
  	the final estimator, while gbtree uses a combination of
    <a style="text-decoration: none" href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html?highlight=decision%20tree#sklearn.tree.DecisionTreeRegressor">Decision Tree Regressors</a>.
    For more details about booster check this <a style="text-decoration: none" href="https://www.avato-consulting.com/?p=28903&lang=en">link</a>
    out.
    </p>"""

    st.markdown(text_9, unsafe_allow_html=True)

    "---"

if page == "Prediction Model":
    "---"

    st.header("Prediction Model")

    text_1 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
    The performance of the best model under test conditions was:
    </p>"""

    st.markdown(text_1, unsafe_allow_html=True)

    metrics = ["RMSE (US$)", "MAE (US$)", "R² (%)"]
    values = [3461.85, 2024.62, 95.68]
    performance = pd.DataFrame(values, index=metrics, columns=["Value"])

    st.table(performance)

    text_2 = """<p style="font-family:Cambria; font-size: 20px; text-align: justify">
    The model can explain about 95.68% of the price variance based on the car's features, showing how robust the model is. Furthermore, the Mean
    Absolute Error of $2024.62 is not even 9% of the average value of all cars; that is, we can be confident that the model is estimating, on
    average, a value very close to the real value of the car. <br><br>Now you can start predicting the price of the cars in the database by
    passing some values to the inputs below!
    </p>"""

    st.markdown(text_2, unsafe_allow_html=True)

    "---"

    # We'll format the inputs for the user and then do a conversion back to the model.

    # --------------------------- Brand ---------------------------#

    col_1, col_2 = st.columns(2)

    new_brand = [
        "Hyundai",
        "Vauxhall",
        "Audi",
        "Volkswagen",
        "Skoda",
        "Mercedes",
        "Toyota",
        "BMW",
        "Ford",
    ]
    old_brand = X_train.brand.unique()
    new_old_brand = dict(zip(new_brand, old_brand))

    brand = col_1.selectbox("Brand", sorted(new_brand), index=4)
    brand = pd.Series(brand).map(new_old_brand)[0]

    X_train = X_train[X_train.brand == brand]

    # --------------------------- Car Model ---------------------------#

    car_model = col_2.selectbox("Model", sorted(X_train.model.unique()))
    X_train = X_train[X_train.model == car_model]

    # ---------------------------- Year -----------------------------#

    year = col_1.selectbox("Year", sorted(X_train.year.unique()))

    # ------------------------- Transmission ------------------------#

    transmission = col_2.selectbox("Transmission", X_train.transmission.unique())

    # -------------------------- Fuel Type --------------------------#

    fuel = col_1.selectbox("Fuel Type", X_train.fuelType.unique())

    # ------------------------- Engine Size -------------------------#

    engine = col_2.selectbox("Engine Size", X_train.engineSize.unique())

    # --------------------------- Mileage ---------------------------#

    mileage = col_1.number_input("Mileage", min_value=0, value=5000, step=50)

    # --------------------------- Predicting ------------------------#

    car = pd.DataFrame([[brand, car_model, year, transmission, mileage, fuel, engine]])
    car.columns = list(X_train.columns)

    model = load("models/model.joblib")

    prediction = model.predict(car)[0]

    # ---------------------------- Profit ---------------------------#

    profit = col_2.number_input(
        "Profit",
        min_value=0.0,
        value=0.05,
        help="Values between 0 and 1 are percentage, and greater than 1 are US Dollars amount!",
    )

    if (profit >= 0) & (profit <= 1):

        bid = prediction * (1 - profit)
        text_profit = "Profit (%)"
        profit *= 100

    elif profit <= round(float(model.predict(car)[0]), 0) - 1:

        bid = prediction - profit
        text_profit = "Profit (US$)"

    "---"

    try:

        info = ["Predicted Price (US$)", text_profit, "Bid (US$)"]
        info_val = [str(int(prediction)), str(round(profit, 2)), str(int(bid))]
        result = pd.DataFrame(info_val, index=info, columns=["Values"])

        st.table(result)

    except NameError:
        st.warning(
            f"The profit should be less than the predicted price of US${int(prediction)}"
        )

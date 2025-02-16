## Student Performance Predictor Dashboard Using Python, and Flask

### About

The project focuses on creating a Student Performance Dashboard using Regression algorithms like Linear Regression, Random Forest, etc., and deploy to a fully functional website using Flask, and HTML. This end-to-end project was inspired by a similar project done by Krish Naik [here](https://www.youtube.com/watch?v=Rv6UFGNmNZg&list=PLZoTAELRMXVPS-dOaVbAux22vzqdgoGhG&index=3). 

### Libraries Used

1. Sckit-learn
2. Pandas
3. Numpy
4. Flask
5. XGBoost

### Installation and Setup

To get started with my repo, first, clone the repository using the commands below

`git clone https://github.com/digs1998/Student-Performance-Predictor-Dashboard.git`

Once you have the repository cloned, create a new environment in Anaconda, using Python 3.10

`conda create -n <env_name> python=3.10 -y`

Run, the installation command below after navigating to the folder.

`cd Student-Performance-Predictor-Dashboard`

`pip install -e .`

This will ensure that the libraries are installed as a package, and are helpful when deploying a machine-learning project to an endpoint.

Now, that you would have all the libraries installed, you can proceed with playing out with the codebase. 

To run the repository, run the command below

`python src/components/data_ingestion.py`

This will perform data ingestion, data transformation, and train the machine learning algorithm as well. 

To deploy the model to a website, I wrote a script in Flask, in `app.py`, which leverages a template from HTML files in the template folder.

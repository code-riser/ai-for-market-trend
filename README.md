<H1>🚀 AI Market Trend Analysis Project</H1>

<P>A beginner-friendly machine learning project that predicts stock market trends using technical indicators and advanced ML algorithms.
</P>

<div><p>📄 Project Report: <a href="https://docs.google.com/document/d/1a3KwjTHALPtBebqicXClC7oHP_J7CAlikjzn6USM3AQ/edit?usp=sharing" alt="Report"> View Report </a></p> 
<p>📊 Live Dashboard: <a href="https://ai-for-trend.streamlit.app/"> Launch Dashboard </a></p>
<p>🎥 Video Demo & Presentation: <a href="https://drive.google.com/file/d/1kL10Dto8SaCSyY7RKwwNlS2L1p4e68nE/view?usp=drivesdk" >Watch Demo </a></p>
</div>
<hr>
<h2>📋 Table of Contents</h2>
<ul>
<li>🎯 <a href="#overview">Overview</a></li>
<li>✨ <a href="#features">Features</a></li>
<li>🚀 <a href="#quick-start">Quick Start</a></li>
<li>📊 <a href="#model-performance">Model Performance</a></li>
<li>🔬 <a href="#technical-details">Technical Details</a></li>
<li>📚 <a href="#learning-resources">Learning Resources</a></li>
<li>🚨 <a href="#Disclaimer">Disclaimer</a></li>
<li>📄 <a href="#license">License</a></li>
<li>🙏 <a href="#acknowledgments">Acknowledgments</a></li>



<div id="overview">
  <h2>🎯 Overview</h2>
  <hr>

  <a href="https://www.youtube.com/" target="_blank" rel="noopener noreferrer">
    Visit Project Demo
  </a>
</div>

<p>
  This project demonstrates how to build an end-to-end AI-powered <strong>Fashion Marketing Trend Analysis</strong> system that combines machine learning, data analytics, and a dynamic web interface to help understand and predict fashion trends.
</p>

<p>
  The system is designed for students and beginners in AI/ML, yet it follows real-world industry practices, making it suitable for academic projects, internships, and portfolio showcase.
</p>

<h4>What it does:</h4>
<div>
  <ul>
    <li>Collects and manages fashion datasets (products, prices, ratings, demand signals)</li>
    <li>Analyzes historical fashion market data to identify trends and patterns</li>
  </ul>

  <hr>

  <ul>
    <li>Trains and evaluates machine learning models for:</li>
    <ul>
      <li>Trend prediction (rising / stable / declining)</li>
      <li>Price movement insights</li>
      <li>Product popularity analysis</li>
    </ul>
  </ul>

  <hr>

  <ul>
    <li>Provides a dynamic, user-friendly web dashboard built with HTML, CSS, and Python (Flask)</li>
    <li>Offers admin-level controls for:</li>
    <ul>
      <li>Dataset management</li>
      <li>Model training and evaluation</li>
      <li>User analytics monitoring</li>
      <li>System activity and performance logs</li>
    </ul>
  </ul>

  <hr>

  <ul>
    <li>Displays fashion products with:</li>
    <ul>
      <li>Product name</li>
      <li>Category</li>
      <li>Price trends</li>
      <li>Ratings & popularity indicators</li>
    </ul>
    <li>Supports scalable architecture with separate modules for data, models, analytics, and logs</li>
  </ul>
</div>

<h3>Key AI Capabilities:</h3>
<ul>
  <li>Feature engineering from fashion market data (price changes, ratings, demand trends)</li>
  <li>Supervised ML models (Random Forest, Logistic Regression, Gradient-based models)</li>
  <li>Model retraining and performance evaluation from the admin panel</li>
  <li>Data-driven insights to support marketing and merchandising decisions</li>
</ul>

<h3>Target Audience:</h3>
<ul>
  <li>Students learning AI, ML, and Data Science</li>
  <li>Beginners building their first full-stack AI project</li>
  <li>Fashion marketing & trend analysis enthusiasts</li>
  <li>Internship, hackathon, and final-year project candidates</li>
</ul>

<p>
  This project focuses on <strong>learning by building</strong>, showing how AI can be applied to real-world fashion marketing problems using clean architecture, modular code, and an intuitive interface.
</p>

<div id="features">
<h2>✨ Features</h2>

<section>
<h3>🔄 Data Collection & Management</h3>

<li>Upload, view, update, and delete fashion datasets (CSV format)</li>

<li>Supports product-level data:</li>

<ul><li>Product name, category, brand</li></ul>

<ul><li>Price history and discount patterns</li></ul>

<ul><li>Ratings, reviews count, and demand signals</li></ul>

<li>Automatic data validation and cleaning</li>

<li>Modular dataset structure for scalability</li>


</section>

<section>

<h3>🔧 Feature Engineering</h3>

<li>Fashion-focused feature extraction:</li>

<ul><li>Price change rate & discount frequency</li></ul>

<ul><li>Rating trends and popularity score</li></ul>

<ul><li>Demand growth/decline indicators</li></ul>

<li>Category-wise trend comparison</li>

<li>Time-based features (daily, weekly, seasonal trends)</li>

<li>Normalization and preprocessing for ML readiness</li>
  
</section>



<section>
  
  <h3>🤖 Machine Learning</h3>

<li>Multiple ML models for trend analysis:</li>

<ul><li>Random Forest</li></ul>

<ul><li>Logistic Regression</li></ul>

<ul><li>Gradient-based models</li></ul>

<li>Trend classification:</li>

<ul><li>Rising Trend</li></ul>

<ul><li>Stable Trend</li></ul>

<ul><li>Declining Trend</li></ul>

<li>Model retraining and evaluation from Admin Panel</li>

<li>Performance metrics tracking (accuracy, precision, recall)</li>

<li>Feature importance analysis for marketing insights</li>
  
</section>
<section>
  
  <h3>📊 Interactive Streamlit Dashboard</h3>
  
<li>Dynamic Visualization product cards with:</li>

<ul><li>Product Selling Prediction</li></ul>

<ul><li> category, and brand</li></ul>

<ul><li>Current price & trend indicator</li></ul>

<ul><li>Customer Segment</li></ul>


</section>



</div>


<div id="quick-start">
<h2>🚀 Quick Start</h2>
  
</div><br>

<div>
  <section>
    <h3>1. Clone & Setup</h3>

  <pre>git clone https://github.com/code-riser/FashionTrend-AI
cd FashionTrend-AI
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
</pre>
<p>This sets up the virtual environment and installs all required Python dependencies.</p>
  </section> <!---end section-->
  
  <section>
    <h3>2. Prepare & Manage Data (Week 1)</h3>
<pre>python data/superstore_sales.csv
</pre>
<li>Loads datasets from the data/ folder</li>

<li>Cleans and validates product, price, rating, and demand data</li>

<li>Prepares data for trend analysis and modeling</li>
  </section> <!---end section-->
  
  <section>
    <h3>3. Feature Engineering & Trend Processing (Week 2)</h3>

<pre>python market_trend_analysis.ipynb

</pre>
<li>Generates fashion-specific features such as:</li>

<ul><li>Price change trends</li></ul>

<ul><li>Popularity and rating movement</li></ul>

<ul><li>Demand growth/decline indicators</li></ul>

<li>Prepares structured data for ML training</li>
  </section> <!---end section-->
  
  <section>
    <h3>4. Train & Evaluate Models (Week 3)</h3>
    <pre>python market_trend_analysis.ipynb

</pre>
<li>Trains machine learning models for fashion trend analysis</li>

<li>Evaluates model performance (accuracy, precision, recall)</li>

<li>Saves trained models in the models/ directory</li>

  </section> <!---end section-->
  
  <section>
    <h3>5.  Product Clustering & Forecasting (Week 4)</h3>

<p> Streamlit Dashboard </p>
<pre>streamlit run app.py
</pre>

  <li>Groups fashion products into trend-based clusters</li>

<li>Forecasts upcoming demand and trend movement</li>
  </section> <!---end section-->
  <section>
    <h3>6. Launch Web Dashboard (Week 4)</h3>

  <pre>python app.py

</pre>


<li>Opens the AI Fashion Trend Dashboard at:
👉 http://localhost:5000</li>
  </section> <!---end section-->


  
</div>

<h2>📁 Project Structure</h2>

<pre>ai-market-trend-analysis/
│
├── data/
│   └── superstore_sales.csv        # Retail sales dataset (Superstore)
│
├── app.py                          # Streamlit web dashboard (AI Market Trend Analysis)
│
├── market_trend_analysis.ipynb     # Jupyter Notebook (EDA + ML model + evaluation)
│
├── requirements.txt                # Python dependencies
│
├── README.md                       # Project documentation
│
└── .gitignore                      # Git ignore rules
</pre>


<div>
<h3>🛠 Installation</h3>
<h4>Prerequisites</h4>
<li>Python 3.8+ (recommended: 3.9 or 3.10)</li>
<li>Git (for cloning the repository)</li>
<li>Internet connection (for data fetching)</li>
<li>Basic knowledge of Python and Jupyter Notebook</li>

<section>
<h4>1. Clone the Repository</h4>

<pre>git clone https://github.com/code-riser/ai-market-trend-analysis.git
cd ai-market-trend-analysis

</pre>

<h4>2. Create a Virtual Environment</h4>
<pre>python -m venv venv
</pre>


<h4>3. Activate the Virtual Environment

On Windows</h4>
<pre>venv\Scripts\activate
</pre>


<h4>On macOS / Linux</h4>
<pre>source venv/bin/activate
</pre>


<h4>4. Install Dependencies</h4>
<pre>pip install -r requirements.txt
</pre>

<pre>
  <h4>5. Verify Project Structure</h4>
  <p>After setup, your project directory should look like this:</p>
</pre>

<pre>ai-market-trend-analysis/
│
├── data/
│   └── superstore_sales.csv
│
├── app.py
├── market_trend_analysis.ipynb
├── requirements.txt
├── README.md
└── .gitignore
</pre>


  <h4>6. Run the Jupyter Notebook (Core Evaluation File)</h4>
  <p>The notebook is the main source of truth for evaluation</p>
<pre>jupyter notebook
</pre>

<h4>Open:</h4>
<pre>market_trend_analysis.ipynb
</pre>
<p>Run all cells top-to-bottom without errors.</p>

<h4>7. Launch the Streamlit Dashboard</h4>
<pre>streamlit run app.py
</pre>
<p>The dashboard will open in your browser at:</p>
<pre>http://localhost:8501
</pre>

<p>✅ Installation Complete

You can now:

Explore data analysis and model training in the notebook

Interact with the dynamic AI-powered dashboard

Analyze sales trends, forecasts, and visual insights</p>


This installs all required libraries for:

<li>Machine Learning models</li>?

<li>Data processing</li>

<li>Flask web application</li>

<li>Dashboard and analytics features</li>
  
</section>


<section>
  <div id="model-performance">
<h3>📊 Model Performance</h3>
<h4>Default Performance (Fashion Market Dataset)</h4>
<h5>(Historical fashion data including prices, ratings, and demand signals)</h5>
  </div>
<section>
  <h2>📊 Model Performance (Fashion Trend Analysis)</h2>

 <table border="1" cellpadding="10" cellspacing="0" style="width:100%; border-collapse: collapse;">
  <thead>
    <tr>
      <th>Model</th>
      <th>R² Score</th>
      <th>MAE (₹)</th>
      <th>Performance Level</th>
      <th>Notes</th>
    </tr>
  </thead>

  <tbody>
    <tr>
      <td>Linear Regression</td>
      <td>0.519</td>
      <td>8,383.77</td>
      <td>Moderate</td>
      <td>Baseline time-series forecasting model</td>
    </tr>

  <tr>
      <td>Linear Regression</td>
      <td>Stable</td>
      <td>Low Error</td>
      <td>Consistent</td>
      <td>Effectively captures long-term sales trends</td>
  </tr>

   <tr>
      <td>Linear Regression</td>
      <td>Fast</td>
      <td>Efficient</td>
      <td>High</td>
      <td>Simple, interpretable, and quick to train</td>
  </tr>

   <tr>
      <td>Linear Regression</td>
      <td>Scalable</td>
      <td>Business-ready</td>
      <td>Reliable</td>
      <td>Suitable for demand forecasting and inventory planning</td>
   </tr>
  </tbody>
</table>

</section>

</section>

<section>

  <h4>📊 Key Metrics</h4>

  <li><b>R² Score (Coefficient of Determination):</b> Measures how well the Linear Regression model explains variance in monthly sales trends.</li>

  <li><b>Mean Absolute Error (MAE):</b> Evaluates the average absolute difference between actual and predicted sales values.</li>

  <li><b>Evaluation Scope:</b> Metrics are computed on historical monthly aggregated sales data.</li>

  <li><b>Forecast Horizon:</b> Next 12 months sales projection based on learned historical trends.</li>

</section>


<section>
  <h4>📈 Feature Usage (Model Inputs)</h4>

  <li><b>Time Index:</b> Sequential numerical index representing monthly progression.</li>

  <li><b>Order Date (Monthly Aggregation):</b> Used to group sales data at month-end frequency.</li>

  <li><b>Sales Amount:</b> Target variable used for trend learning and forecasting.</li>

  <li><b>Derived Time Features:</b> Year and Month extracted for exploratory analysis.</li>

  <p><i>Note:</i> Feature importance ranking is not applied as the project uses a univariate Linear Regression forecasting approach.</p>

</section>


<section>
  <div id="technical-details">
  <h3>🔬 Technical Details</h3>
  </div>
  <h4>Data Pipeline</h4>

  <li><b>Collection:</b> CSV-based retail sales dataset (Superstore Sales).</li>

  <li><b>Loading:</b> Data loaded using pandas with latin1 encoding and parsed order dates.</li>

  <li><b>Cleaning:</b> Removal of missing or invalid sales records.</li>

  <li><b>Transformation:</b> Monthly aggregation of sales using time-based grouping.</li>

  <li><b>Feature Engineering:</b> Time index creation for regression modeling.</li>

  <li><b>Training:</b> Linear Regression model trained on historical monthly sales.</li>

  <li><b>Evaluation:</b> R² Score and MAE used to assess prediction quality.</li>

</section>


<section>
  <h3>📉 Time-Series Trend Analysis Implemented</h3>

  <h4>Sales Trend Indicators</h4>

  <li>Monthly total sales movement over time</li>

  <li>Growth and decline phases in historical retail demand</li>

  <li>Pattern recognition in seasonal and long-term sales behavior</li>

  <h4>Forecasting Indicators</h4>

  <li>Linear trend continuation based on historical data</li>

  <li>Future monthly sales estimation (12-month horizon)</li>

  <li>Comparison of actual vs predicted sales trends</li>

</section>


<section>
  <h3>🧠 Model Architecture</h3>

  <h4>Linear Regression (Forecasting Model)</h4>

  <pre>
LinearRegression()
  </pre>

  <li>Supervised machine learning model for numerical prediction</li>

  <li>Trained on time-indexed monthly sales data</li>

  <li>Chosen for simplicity, interpretability, and educational clarity</li>

</section>


<hr>

<div id="learning-resources">
<h2>📚 Learning Resources</h2>
<ul>
<li>Pandas / Numpy – Data processing</li>
<li>Scikit-learn – Machine learning</li>
<li>Streamlit – Dashboard & frontend</li>
<li>Plotly / Matplotlib – Data visualization</li>
<li>Public Superstore sales datasets for practice</li>
</ul>
</div>

<hr>


<section>
  <div id="Disclaimer">
  <h3>🚨 Disclaimer</h3>
  </div>
  <h4>IMPORTANT:</h4>
  <p>This project is intended strictly for educational and learning purposes.</p>

  <li>📚 Educational Use Only: Demonstrates AI, machine learning, and data analytics concepts using retail sales data.</li>

  <li>❌ Not Business or Financial Advice: Forecasts should not be used for real-world commercial or investment decisions.</li>

  <li>📊 Historical Data Limitation: Predictions rely solely on past sales patterns and assume trend continuity.</li>

  <li>🎯 Model Constraints: Linear Regression provides baseline forecasting accuracy suitable for learning, not production deployment.</li>

  <li>⚠️ Market Uncertainty: Real-world sales are influenced by external factors beyond historical data.</li>

  <h4>Legal Notice</h4>
  <p>The authors and contributors are not responsible for any business or financial outcomes resulting from the use of this project.</p>
<div id="license">
  <h4>📄 License</h4>
  <p>This project is licensed under the MIT License. See the LICENSE file for details.</p>
  </div>

</section>

<div id="acknowledgments">
<h4>🙏 Acknowledgments</h4>
</div>
<li>Open-source fashion and e-commerce datasets used for learning purposes</li>

<li>scikit-learn for reliable machine learning implementations</li>

<li>python for backend web development</li>

<li>Streamlit for frontend design</li>

<li>Matplotlib & Plotly for data visualization tools</li>








  <h4>📞 Support</h4>
  <p>For queries, feedback, or academic guidance:</p>
  <li>📧 Contact: mdsahidalam7860@gmail.com</li>
</section>
 
</div>

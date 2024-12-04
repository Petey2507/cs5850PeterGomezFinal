import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import matplotlib.pyplot as plt

california_data = fetch_california_housing()
data = pd.DataFrame(california_data.data, columns=california_data.feature_names)
data['MedHouseVal'] = california_data.target

X = data[['MedInc', 'HouseAge', 'AveRooms', 'Latitude', 'Longitude', 'Population']]
y = data['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

###############################
# OLS Linear Regression Section
model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
print(model.summary())

# MedInc vs MedHouseVal
data['MedIncBin'] = pd.qcut(data['MedInc'], q=20)
bin_means = data.groupby('MedIncBin')['MedHouseVal'].mean()
bin_centers = data.groupby('MedIncBin')['MedInc'].mean()
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, bin_means, marker='o', linestyle='-')
plt.xlabel("Median Income")
plt.ylabel("Average Median House Value")
plt.title("Median Income vs. Average Housing Prices")
plt.show()

# AveRooms vs MedHouseVal
data['AveRoomsBin'] = pd.qcut(data['AveRooms'], q=20)
bin_means = data.groupby('AveRoomsBin')['MedHouseVal'].mean()
bin_centers = data.groupby('AveRoomsBin')['AveRooms'].mean()
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, bin_means, marker='o', linestyle='-')
plt.xlabel("AveRooms")
plt.ylabel("Average Median House Value")
plt.title("AveRooms vs. Average Housing Prices")
plt.show()

# Population vs MedHouseVal
data['PopulationBin'] = pd.qcut(data['Population'], q=20)
bin_means = data.groupby('PopulationBin')['MedHouseVal'].mean()
bin_centers = data.groupby('PopulationBin')['Population'].mean()
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, bin_means, marker='o', linestyle='-')
plt.xlabel("Population")
plt.ylabel("Average Median House Value")
plt.title("Population vs. Average Housing Prices")
plt.show()

# HouseAge vs MedHouseVal
data['HouseAgeBin'] = pd.qcut(data['HouseAge'], q=20, duplicates='drop')
bin_means = data.groupby('HouseAgeBin')['MedHouseVal'].mean()
bin_centers = data.groupby('HouseAgeBin')['HouseAge'].mean()
plt.figure(figsize=(10, 6))
plt.plot(bin_centers, bin_means, marker='o', linestyle='-')
plt.xlabel("HouseAge")
plt.ylabel("Average Median House Value")
plt.title("HouseAge vs. Average Housing Prices")
plt.show()

# Get MSE
y_pred = model.predict(sm.add_constant(X_test))
mse = np.mean((y_test - y_pred) ** 2)
print(f"Mean Squared Error on test set: {mse:.4f}")

######################################
# Decision Tree Classification Section
data['MedHouseValClass'] = pd.qcut(data['MedHouseVal'], q=3, labels=['Low', 'Medium', 'High'])
X = data[['MedInc', 'HouseAge', 'AveRooms', 'Latitude', 'Longitude', 'Population']]
y = data['MedHouseValClass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = tree.DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Show accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Show tree
plt.figure(figsize=(25, 20))
tree.plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True, max_depth=3)
plt.show()

#############################
# Feature importance analysis
importances = clf.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance in Decision Tree')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

###################################################
# Confusion Matrix for Decision Tree Classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class_order = ['Low', 'Medium', 'High']
cm = confusion_matrix(y_test, y_pred, labels=class_order)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_order)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

#######################
# Classification Report
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred, target_names=class_order)
print("Classification Report:")
print(report)

##########################
# Dynamic Heat Map Section
import folium
from folium.plugins import HeatMap, MarkerCluster
import geopandas as gpd

data['Coordinates'] = list(zip(data['Latitude'], data['Longitude']))
data['IncomeCategory'] = pd.qcut(data['MedInc'], q=3, labels=['Low', 'Medium', 'High'])
data['HousePriceCategory'] = pd.qcut(data['MedHouseVal'], q=3, labels=['Low', 'Medium', 'High'])

income_heatmap_data = {
    category: [
        [float(lat), float(lon), float(medinc)]
        for lat, lon, medinc in data[data['IncomeCategory'] == category][['Latitude', 'Longitude', 'MedInc']].values.tolist()
    ]
    for category in ['Low', 'Medium', 'High']
}

house_price_heatmap_data = {
    category: [
        [float(lat), float(lon), float(price)]
        for lat, lon, price in data[data['HousePriceCategory'] == category][['Latitude', 'Longitude', 'MedHouseVal']].values.tolist()
    ]
    for category in ['Low', 'Medium', 'High']
}

california_center = [36.7783, -119.4179]
california_map = folium.Map(location=california_center, zoom_start=6, tiles='OpenStreetMap')

try:
    california_shape = gpd.read_file('C:/Users/link2/Desktop/CS5850/Project/California_State_Boundary.shp')
    folium.GeoJson(california_shape, name='California Boundary').add_to(california_map)
except Exception as e:
    print(f"Error loading shapefile: {e}")

# Heatmaps
try:
    for category, data_points in income_heatmap_data.items():
        income_layer = folium.FeatureGroup(name=f'{category} Income Heatmap')
        HeatMap(data_points, radius=10, blur=15, max_zoom=1).add_to(income_layer)
        income_layer.add_to(california_map)
except Exception as e:
    print(f"Error adding income heatmaps: {e}")

try:
    for category, data_points in house_price_heatmap_data.items():
        price_layer = folium.FeatureGroup(name=f'{category} House Price Heatmap')
        HeatMap(data_points, radius=10, blur=15, max_zoom=1).add_to(price_layer)
        price_layer.add_to(california_map)
except Exception as e:
    print(f"Error adding house price heatmaps: {e}")

# Clusters
try:
    for category in ['Low', 'Medium', 'High']:
        cluster = MarkerCluster(name=f'{category} Income Clusters').add_to(california_map)
        category_data = data[data['IncomeCategory'] == category]
        for _, row in category_data.iterrows():
            latitude = float(row['Latitude'])
            longitude = float(row['Longitude'])
            popup = f"Income: ${float(row['MedInc']*10000):.2f}, Population: {int(row['Population'])}"
            folium.Marker(location=[latitude, longitude], popup=popup, icon=folium.Icon(color='red')).add_to(cluster)
except Exception as e:
    print(f"Error adding income clusters: {e}")

try:
    for category in ['Low', 'Medium', 'High']:
        cluster = MarkerCluster(name=f'{category} House Price Clusters').add_to(california_map)
        category_data = data[data['HousePriceCategory'] == category]
        for _, row in category_data.iterrows():
            latitude = float(row['Latitude'])
            longitude = float(row['Longitude'])
            popup = f"House Price: ${float(row['MedHouseVal']*100000):.2f}, Population: {int(row['Population'])}"
            folium.Marker(location=[latitude, longitude], popup=popup, icon=folium.Icon(color='blue')).add_to(cluster)
except Exception as e:
    print(f"Error adding house price clusters: {e}")

# Legend
legend_html = '''
<div style="position: fixed; bottom: 50px; left: 50px; width: 250px; height: 200px;
            background-color: white; z-index:9999; font-size:14px; padding: 10px; border:2px solid grey;">
<strong>Legend:</strong><br>
<i style="background: red; width: 10px; height: 10px; display: inline-block;"></i> Income Clusters<br>
<i style="background: blue; width: 10px; height: 10px; display: inline-block;"></i> House Price Clusters<br>
</div>
'''
california_map.get_root().html.add_child(folium.Element(legend_html))

folium.LayerControl().add_to(california_map)

try:
    california_map.save('california_categorized_visualization.html')
    print("Map saved successfully as 'california_categorized_visualization.html'")
except Exception as e:
    print(f"Error saving map: {e}")

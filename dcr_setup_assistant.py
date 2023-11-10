import io
import streamlit as st
import snowflake_dcr as dcr
import os
from zipfile import ZipFile
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from IPython.display import Image
from sklearn.tree import export_graphviz ,plot_tree
import pydotplus
import plotly.express as px
import seaborn as sns
from io import BytesIO
# Page settings
st.set_page_config(
    page_title="Vizuara AI Learning",
    page_icon="❄️🚢",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Survive the Titanic: Unraveling the Mystery with Machine Learning Magic!"
    }
)
# Set up main page
col1, col2 = st.columns((6, 1))
col1.title("🏔 🚢Titanic Adventure: Unraveling Secrets with Machine Learning Magic❄️ 🚢")
col2.image("assets/snowflake_dcr_multi.png", width=120)
st.sidebar.image("assets/bear_snowflake_hello.png")
action = st.sidebar.radio("What action would you like to take?", ("Unveiling the  Dataset🐻‍❄",
                                                                  "Navigating Data Waters 🐧️",
                                                                  "Shining in Darkness. ☃️",
                                                                  "Casting the Net 🏂🏽",
                                                                  "Engines of Prediction: Constructing the Model ⚓",
                                                                  "Safe Harbors: Concluding the Predictive Odyssey 🌊"))

st.markdown("Ahoy, fellow data explorers! 🚢✨")
st.markdown("Let's set sail on a Titanic adventure, where the waves of history crash against the iceberg of mystery.")
st.markdown("Picture this: April 15, 1912, the 'unsinkable' RMS Titanic takes a plunge after an encounter with an iceberg.")
st.markdown("Tragically, not enough lifeboats, and 1502 souls out of 2224 bid farewell to the Titanic voyage.")
st.markdown("But fear not, intrepid coders! Luck played its part, and now, with your wit and wizardry, we're on a quest.")
st.markdown("Who were the chosen ones? What magical traits increased the odds of survival?")
st.markdown("Join us in this coding escapade as we summon the predictive model genie to answer: 'Who sails through the storm, and who succumbs to the waves?' 🌊🔮")


dcr_data_options = ['Media & Advertising', 'None']

# Create dcr object
data_clean_room = dcr.SnowflakeDcr()
zip_buffer = io.BytesIO()



# Gets version info
def get_version_info(repo_path):
    version_file = repo_path + "VERSION.md"
    version_list = []
    # Read version file
    try:
        with open(version_file, "r", encoding='utf-8') as fin:
            for line in fin:
                version_list.append(line)
    finally:
        return version_list


# Used to load the zip file buffer in-memory
def load_zip_buffer(snowflake_dcr, buffer, do_include_comments):
    # Add script number to help sort and show the correct order
    script_number = 1
    if do_include_comments:
        with ZipFile(buffer, "w") as archive:
            for script in snowflake_dcr.prepared_script_dict.keys():
                # Remove path
                script_name = script.lstrip("/")

                # Add number
                script_name = str(script_number) + " - " + script_name
                script_number += 1

                archive.writestr(script_name, snowflake_dcr.prepared_script_dict[script])
    else:
        with ZipFile(buffer, "w") as archive:
            for script in snowflake_dcr.cleaned_script_dict.keys():
                # Remove path
                script_name = script.lstrip("/")

                # Add number
                script_name = str(script_number) + " - " + script_name
                script_number += 1

                # write to archive
                archive.writestr(script_name, snowflake_dcr.cleaned_script_dict[script])


path = os.getcwd() + "/data-clean-room/"

@st.cache_data
def get_dataset():

    try:
        dataset = pd.read_csv('train.csv')
    except:
        st.warning("Wrong file")

    return dataset
dataset = get_dataset()
data_attributes = {
    "Survival": {
        "Description": "Shows if the passenger survived or not. 1 stands for survived and 0 stands for not survived.",
        "Type": "int",
        "Value Range": "0 = No, 1 = Yes",
        "Value Average": np.mean(dataset["Survived"])
    },
    "Pclass": {
        "Description": "Ticket class. 1 stands for First class ticket. 2 stands for Second class ticket. 3 stands for Third class ticket.",
        "Type": "int",
        "Value Range": "1 = 1st, 2 = 2nd, 3 = 3rd",
        "Value Average": np.mean(dataset["Pclass"])
    },
    "Gender": {
        "Description": "Passenger's Gender. It's either Male or Female.",
        "Type": "string",
        "Value Range": "Male, Female",
        "Value Average": "N/A"
    },
    "Age": {
        "Description": "Passenger's age. NaN values in this column indicates that the age of that particular passenger has not been recorded.",
        "Type": "float",
        "Value Range": "{}-{}".format(np.min(dataset["Age"]), np.max(dataset["Age"])),
        "Value Average": np.mean(dataset["Age"])
    },
    "SibSp": {
        "Description": "Number of siblings or spouses travelling with each passenger.",
        "Type": "int",
        "Value Range": "{}-{}".format(np.min(dataset["SibSp"]), np.max(dataset["SibSp"])),
        "Value Average": np.mean(dataset["SibSp"])
    },
    "Parch": {
        "Description": "Number of parents of children travelling with each passenger.",
        "Type": "int",
        "Value Range": "{}-{}".format(np.min(dataset["Parch"]), np.max(dataset["Parch"])),
        "Value Average": np.mean(dataset["Parch"])
    },
    "Ticket": {
        "Description": "Ticket number",
        "Type": "string",
        "Value Range": "N/A",
        "Value Average": "N/A"
    },
    "Fare": {
        "Description": "How much money the passenger has paid for the travel journey",
        "Type": "float",
        "Value Range": "{:.2f}-{:.2f}".format(np.min(dataset["Fare"]), np.max(dataset["Fare"])),
        "Value Average": np.mean(dataset["Fare"])
    },
    "Cabin": {
        "Description": "Cabin number of the passenger. NaN values in this column indicates that the cabin number of that particular passenger has not been recorded.",
        "Type": "string",
        "Value Range": "N/A",
        "Value Average": "N/A"
    },
    "Embarked": {
        "Description": "Port from where the particular passenger was embarked/boarded.",
        "Type": "string",
        "Value Range": "C = Cherbourg, Q = Queenstown, S = Southampton",
        "Value Average": "N/A"
    }
}
survived_count = dataset[dataset["Survived"] == 1].groupby("Gender")["Survived"].count()
not_survived_count = dataset[dataset["Survived"] == 0].groupby("Gender")["Survived"].count()
survived = dataset[dataset['Survived'] == 1]
not_survived = dataset[dataset['Survived'] == 0]
dataset["Gender"] = dataset["Gender"].map({'male': 0, 'female': 1})
X_final = dataset[['Gender','Pclass','Fare','Embarked']]
y_final = dataset[['Survived']]
# Build form based on selected action
if action == "Unveiling the  Dataset🐻‍❄":
    st.subheader("❄️  Aboard the Titanic: Unveiling the Titanic Data! ❄️")
    st.markdown("Embark on a journey into the heart of the Titanic disaster dataset. 🚢💻 Explore the stories hidden in the numbers – from passenger class and age to gender and fare.")
    st.markdown("Each data point holds a piece of the puzzle, and together, we're about to decipher its secrets.")
    if st.checkbox('Show data'):
        st.success("Here comes the dataset successfully")
        st.write(dataset.head())
    # with st.form("initial_deployment_form"):

        # st.subheader('Display the dataset')

        # display_data_attributes = st.checkbox("Show Data Attributes", value=True)    


        # dcr_version = "DCR 5.5 General Availability"
        # abbreviation = st.text_input("What database abbreviation would you like? (Leave blank for default)")
        # provider_account = st.text_input(label="What is the Provider's account identifier?",
        #                                  help="This should be just the account locator.  Anything beyond a '.' will be "
        #                                       "removed automatically.")
        # consumer_account = st.text_input(label="What is the Consumer's account identifier?",
        #                                  help="This should be just the account locator.  Anything beyond a '.' will be "
        #                                       "removed automatically.")
        # dcr_data_selection = st.selectbox("Would you like to load demo data?", dcr_data_options)
        # include_comments = st.checkbox("Include comments in scripts", True)

        # submitted = st.form_submit_button("Run")

        # if submitted:
        #     if provider_account == consumer_account:
        #         st.error("Provider and consumer cannot be the same account!")
        #     else:
        #         with st.spinner("Generating Clean Room Scripts..."):
        #             data_clean_room.prepare_dcr_deployment(True, dcr_version, provider_account, None, consumer_account,
        #                                                    None, abbreviation, path, dcr_data_selection)
        #             data_clean_room.execute()

        #         # Message dependent on debug or not
        #         st.success("Scripts Ready for Download!")

        #         # Populate zip buffer for download buttons
        #         load_zip_buffer(data_clean_room, zip_buffer, include_comments)
        #         st.snow()

elif action == "Navigating Data Waters 🐧️":
    # Form for adding consumers
    st.subheader("❄️ Distinct features within the data! ❄️")
    st.markdown("""
    Now, as we sail into the vast sea of data, let's familiarize ourselves with the types of information onboard.
    """)

    st.markdown("""
    Prepare to navigate through the numerical depths, ride the categorical currents, and surf the textual tides.
    """)

    st.markdown("""
    Witness the intricate transformation of raw data, unfolding its secrets and paving the way for our exciting predictive journey.
    """)
    selected_attribute = st.selectbox("Select an Attribute", list(data_attributes.keys()))

    if selected_attribute in data_attributes:
        # Display data attributes for the selected attribute
        data_attr_table = pd.DataFrame(data_attributes[selected_attribute], index=[selected_attribute])
        st.table(data_attr_table)
    # with st.form("additional_consumer_form"):
    #     dcr_version = "DCR 5.5 General Availability"
    #     abbreviation = st.text_input("What database abbreviation does the Provider have? (Leave blank for default)")
    #     provider_account = st.text_input(label="What is the existing Provider's account identifier?",
    #                                      help="This should be just the account locator.  Anything beyond a '.' will be "
    #                                           "removed automatically.")
    #     consumer_account = st.text_input(label="What is the new Consumer's account identifier?",
    #                                      help="This should be just the account locator.  Anything beyond a '.' will be "
    #                                           "removed automatically.")
    #     dcr_data_selection = st.selectbox("Would you like to load demo data?", dcr_data_options)
    #     include_comments = st.checkbox("Include comments in scripts", True)

    #     submitted = st.form_submit_button("Run")

    #     if submitted:
    #         if provider_account == consumer_account:
    #             st.error("Provider and consumer cannot be the same account!")
    #         else:
    #             path = os.getcwd() + "/data-clean-room/"

    #             with st.spinner("Generating Clean Room Scripts..."):
    #                 data_clean_room.prepare_consumer_addition(True, dcr_version, provider_account,
    #                                                           None, consumer_account, None,
    #                                                           abbreviation, path, dcr_data_selection)
    #                 data_clean_room.execute()

    #             st.success("Scripts Ready for Download!")

    #             # Populate zip buffer for download buttons
    #             load_zip_buffer(data_clean_room, zip_buffer, include_comments)
    #             st.snow()
        st.subheader("🐼 Understanding types of data! 🐼" )

    st.write("In machine learning and data analysis, it's important to distinguish between two main types of data:")
    st.markdown("1. **Categorical Data**: Categorical data represents discrete and distinct categories or labels. It can be further divided into nominal and ordinal data. Nominal data doesn't have any inherent order, while ordinal data has a specific order or ranking.")

    st.markdown("2. **Non-categorical data**, as opposed to categorical data, represents values on a continuous scale. This type of data is typically measured and can take any real value within a certain range. Examples include numerical data such as temperature, height, or weight")

# Label Encode Categorical Variable
    if True:

        encode_option = st.radio("Select an Attribute for Categorical data", ["Gender", "PassengerId", "Fare",'Survival','Pclass','Age','SibSp','Parch','Ticket','Cabin','Embarked'], index=None)
        
        if encode_option == "Gender":
            # Perform label encoding on the "Gender" column
            
            label_encoding_result = pd.DataFrame({
                "Original Gender": ['male', 'female'],
                "Encoded Value": [0, 1]
            })
            st.write(dataset["Gender"].head())
            # Display the mapping results to the user
            # st.write("Label Encoding Results:")
            # st.table(label_encoding_result)
            st.success("You successfully selected Categorical data")
                       
        elif encode_option == "PassengerId":
            st.write(dataset["PassengerId"].head())
            st.warning("PassengerID should not be encoded. It is a unique identifier for each passenger and does not carry meaningful information for the model. Encoding it could lead to confusion and misinterpretation of the data. Please select 'Gender' or 'Fare' and retry.")
        elif encode_option == "Fare":
            st.write(dataset["Fare"].head())
            st.warning("Fare should not be encoded. It represents the ticket price, which is a continuous numerical value. Encoding it as a category would distort its meaningful numerical relationship and potentially lead to erroneous interpretations by the model. Leave 'Fare' as a numerical feature to allow the model to analyze it appropriately. Please select 'Gender' or 'PassengerId' and retry.")
        elif encode_option == "Age":
            st.write(dataset["Age"].head())
            st.warning("Age is a numerical attribute representing the passenger's age. It should not be encoded as a category. Leave 'Age' as a numerical feature for the model to analyze it appropriately.")

        elif encode_option == "SibSp":
            st.write(dataset["SibSp"].head())
            st.warning("SibSp is a numerical attribute representing the number of siblings or spouses traveling with each passenger. It should not be encoded as a category. Leave 'SibSp' as a numerical feature for the model to analyze it appropriately.")

        elif encode_option == "Parch":
            st.write(dataset["Parch"].head())
            st.warning("Parch is a numerical attribute representing the number of parents or children traveling with each passenger. It should not be encoded as a category. Leave 'Parch' as a numerical feature for the model to analyze it appropriately.")
        elif encode_option == "Ticket":
            st.write(dataset["Ticket"].head())
            st.warning("Ticket is a non-categorical attribute representing the ticket number. It should not be encoded as a category. Leave 'Ticket' as a non-categorical feature for the model to analyze it appropriately.")

        elif encode_option == "Cabin":
            st.write(dataset["Cabin"].head())
            st.warning("Cabin is a non-categorical attribute representing the cabin number. It should not be encoded as a category. Leave 'Cabin' as a non-categorical feature for the model to analyze it appropriately.")

elif action == "Shining in Darkness. ☃️":
    # Form for adding providers
    st.markdown("Dive into the depths of data visualization to illuminate the patterns of survival.")

    st.subheader("❄️  Visualizing Survival Patterns! ❄️")
    st.markdown("Witness the movement of sea and flow of features like age, class, and embarkation point.")
    st.markdown("Visual cues will guide us towards understanding the stark realities faced by passengers on that fateful night.")

    display_options = ["Select an option", "Gender vs. Survival", "Pclass vs. Survival", 
                    "Pclass & Gender vs. Survival", "Pclass, Gender & Embarked vs. Survival", 
                    "Embarked Bar Plot", "Parch vs Survival", "SibSp vs Survival", "Age vs. Survival"]

    selected_option = st.selectbox("Choose a visualization", display_options, key="selected_option")

    if selected_option == "Gender vs. Survival":
        st.subheader("Gender vs. Survival")

        total_passengers_by_gender = dataset['Gender'].value_counts()
        survived_count = dataset[dataset['Survived'] == 1]['Gender'].value_counts()
        not_survived_count = dataset[dataset['Survived'] == 0]['Gender'].value_counts()

        survived_percentage = (survived_count / total_passengers_by_gender) * 100
        not_survived_percentage = (not_survived_count / total_passengers_by_gender) * 100

        # Create a color palette
        palette = sns.color_palette("Set2")

        # Plot for survived passengers by gender
        fig_survived, ax_survived = plt.subplots(figsize=(8, 6))
        sns.barplot(x=survived_percentage.index.map({0: "Male", 1: "Female"}), y=survived_percentage.values, palette=palette)
        ax_survived.set_xlabel("Gender")
        ax_survived.set_xlabel("Gender (0: Male, 1: Female)")
        ax_survived.set_ylabel("Survival Percentage")
        ax_survived.set_title("Survived Passengers by Gender")
        st.pyplot(fig_survived)

        # Plot for Male passenFemale by gender
        fig_not_survived, ax_not_survived = plt.subplots(figsize=(8, 6))
        sns.barplot(x=not_survived_percentage.index.map({0: "Male", 1: "Female"}), y=not_survived_percentage.values, palette=palette)
        ax_not_survived.set_xlabel("Gender (0: Male, 1: Female)")
        ax_not_survived.set_ylabel("Not Survival Percentage")
        ax_not_survived.set_title("Not Survived Passengers by Gender")
        st.pyplot(fig_not_survived)
        
        gender_survived = dataset.groupby('Gender').Survived.value_counts(normalize=True).unstack(level=0) * 100  # Convert to percentages

        # Create a color palette
        palette = sns.color_palette("Set2")

        # Rename the columns to "Male" and "Female"
        gender_survived.columns = ["Females", "Males"]
        gender_survived.index = ["Survived", "Not Survived"]


        # Plot for survived passengers by gender
        fig_gender_survive, ax = plt.subplots()
        gender_survived.plot(kind='bar', ax=ax)
        ax.set_title('Survived Passengers by Gender')
        
        ax.set_xticklabels(gender_survived.index, rotation=0, horizontalalignment='center')

        st.pyplot(fig_gender_survive)
        
        st.subheader("Question on Gender vs Survival:")
        st.write("Based on the provided summary, which gender has a better survival chance?")

        # Create options for the MCQ
        options = ["Males", "Females"]

        # Get the user's selection
        user_selection = st.selectbox("Select an option:", options, index=None)

        # Correct answer
        correct_answer = "Females"

        # Check if the user's selection is correct and provide feedback
        if user_selection is not None:
            if user_selection == correct_answer:
                st.success("Correct! Females have better survival chance.")
                st.success("Note: Females have better survival chance.")
            else:
                st.error("Incorrect. Hint: Could you try to find the summary of the graph ?")
    # Display the selected plots based on the user's choices

    elif selected_option == "Pclass vs. Survival":
        
        st.subheader("Pclass vs. Survival")

        # Calculate percentages for survived passengers by Pclass
        pclass_survived = dataset.groupby('Pclass').Survived.value_counts(normalize=True).unstack(level=0) * 100

        # Rename the x-axis labels
        pclass_survived.columns = ['First Class Ticket', 'Second Class Ticket','Third Class Ticket']
        pclass_survived.index=['Not Survived','Survived']
        # Create a bar plot for Pclass vs. Survival
        fig_pclass_survive, ax = plt.subplots()
        pclass_survived.plot(kind='bar', ax=ax)
        ax.set_xlabel('Pclass')
        ax.set_xticklabels(pclass_survived.index, rotation=0, horizontalalignment='center')
        ax.set_ylabel('Percentage')
        ax.set_title('Survived Passengers by Pclass')
        st.pyplot(fig_pclass_survive)
        

        # Calculate average survival rate by Pclass
        pclass_survived_average = dataset[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()

        # Rename the x-axis labels
        pclass_survived_average['Pclass'] = pclass_survived_average['Pclass'].map({1: 'First Class', 2: 'Second Class', 3: 'Third Class'})

        # Create a bar plot for average survival rate by Pclass
        fig_pclass_survived_avg, ax = plt.subplots()
        pclass_survived_average.plot(kind='bar', ax=ax, x='Pclass', y='Survived')
        ax.set_xlabel('Pclass')
        ax.set_ylabel('Average Survival Rate')
        
        ax.set_xticklabels(pclass_survived_average['Pclass'], rotation=0, horizontalalignment='center')

        st.pyplot(fig_pclass_survived_avg)
        
        # Display the MCQ question
        st.subheader("Question Pclass vs Survival")
        st.write("Based on the provided summaries, what can be concluded about the relationship between passenger class (Pclass) and survival on the Titanic?")

        # Create options for the MCQ
        options = [
            "Higher class passengers have a higher survival chance.",
            "Lower class passengers have a higher survival chance.",
            "There is no clear relationship between Pclass and survival.",
            "Pclass does not impact survival."
        ]

        # Get the user's selection
        user_selection = st.radio("Select the correct statement:", options, index=None)  # Set index to None

        # Correct answer
        correct_answer = "Higher class passengers have a higher survival chance."

        # Check if the user's selection is correct and provide feedback
        if user_selection is not None:
            if user_selection == correct_answer:
                st.success("Correct! Higher class passengers have a higher survival chance.")
                st.success("Note: Higher class passengers have better survival chance (may be because they are more privileged to be saved).")
                st.success("Note: Higher class passengers (low Pclass) have better average survival than the low class (high Pclass) passengers.")
            else:
                st.error("Incorrect. Hint: Could you try to find out the summary of the graph ?")



    elif selected_option == "Pclass & Gender vs. Survival":
        
        st.subheader("Pclass & Gender vs. Survival")
        tab = pd.crosstab(dataset['Pclass'], dataset['Gender'])
        percent_tab = (tab.div(tab.sum(1).astype(float), axis=0) * 100).round(1)  # Convert to percentages and round to one decimal place

        # Define custom tick labels for the x-axis
        custom_labels = ['First Class', 'Second Class', 'Third Class']

        # Plot the data
        fig, ax = plt.subplots()
        percent_tab.plot(kind="bar", stacked=False, ax=ax)

        # Set custom tick labels for the x-axis
        ax.set_xticklabels(custom_labels)

        # Set labels and title
        plt.legend(['Male', 'Female'])
        plt.xlabel('Pclass')
        plt.ylabel('Percentage (%)')
        plt.title('Survived Passengers by Pclass')

        # Display the modified plot using st.pyplot
        # st.pyplot(fig)
        

        factor_plot = sns.catplot(x="Gender", y="Survived", hue='Pclass', size=40, aspect=5 , data=dataset,height=100)
        # st.pyplot(factor_plot)
        dataset['Gender'] = dataset['Gender'].map({0: 'Female', 1: 'Male'})
        sns.set(style="whitegrid")
        plt.figure(figsize=(400, 200))
        g = sns.catplot(x="Gender", y="Survived", hue="Pclass", kind="bar", col="Pclass", data=dataset)
        g.set_axis_labels("Gender", "Survived")
        st.pyplot(plt.gcf())
        st.write("This is a catplot showing the relationship between Gender, Survived, and Pclass.")

        # Convert the notes into st.success messages

        st.write("Question:")
        st.write("What can be observed from the catplot showing the relationship between Gender, Survived, and Pclass?")

        options = ["A) Women from all Pclasses have almost 100% survival chance.",
                "B) Women from 1st and 2nd Pclass have almost 100% survival chance.",
                "C) Men from 1st and 2nd Pclass have almost 100% survival chance.",
                "D) Men from 2nd and 3rd Pclass have almost 100% survival chance."]

        # Let the user select an option
        selected_option = st.radio("Select an option:", options,index=None)
        dataset['Gender'] = dataset['Gender'].map({0: 'Female', 1: 'Male'})
        # Provide feedback based on the selected option
        if selected_option is not None:
            if selected_option == "B) Women from 1st and 2nd Pclass have almost 100% survival chance.":
                st.success("Correct! Women from 1st and 2nd Pclass have almost 100% survival chance.")
                st.success("NOTE:")
                st.success("- Women from 1st and 2nd Pclass have almost 100% survival chance.")
                st.success("- Men from 2nd and 3rd Pclass have only around 10% survival chance.")
            else:
                st.error("Incorrect. Hint: Could you try to find the summary of the graph ?")

    elif selected_option == "Pclass, Gender & Embarked vs. Survival":
        
        st.subheader('Pclass, Gender & Embarked vs. Survival')
        dataset['Gender'] = dataset['Gender'].map({0: 'Male', 1: 'Female'})

        st.set_option('deprecation.showPyplotGlobalUse', False)

        sns.set(style="whitegrid")
        g = sns.catplot(x="Pclass", y="Survived", hue="Gender", col="Embarked", kind="bar", data=dataset)

        # Set axis labels
        g.set_axis_labels("Pclass", "Survived")
        embarked_mapping = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}

        for ax, embarked_value in zip(g.axes.flat, dataset['Embarked'].unique()):
            ax.set_title(f"Embark: {embarked_mapping.get(embarked_value)}")
        # Replace the column descriptions with "First Class", "Second Class", and "Third Class"
        g.set_xticklabels(['First Class', 'Second Class', 'Third Class'])

        st.pyplot(plt.gcf())

        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x="Embarked", y="Survived", data=dataset)

        # Set axis labels and title
        plt.xlabel("Embarked")
        plt.ylabel("Survived (%)")  # Update the y-axis label
        plt.title("Survived Passengers by Embarked")

        # Format the y-axis labels as absolute percentages
        ax.set_yticklabels([f"{int(y*100)}%" for y in ax.get_yticks()])
        embarked_mapping = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
        ax.set_xticklabels(['Cherbourg', 'Queenstown', 'Southampton'])
        question = f"If you are Male, which port should you NOT embark from?"
        st.pyplot(plt)
        # Provide answer options
        options = ['Cherbourg', 'Queenstown', 'Southampton']

        # Correct answer
        correct_answer = 'Queenstown'

        # Display the question and radio options
        user_response = st.radio(question, options)

        # Check the user's answer
        if user_response == correct_answer:
            st.success("Correct! Queenstown is the correct answer.")
        else:
            st.error(f"Sorry, that's incorrect. Hint: Please Check Queenstown Embark graph")

        # Display the bar plot using st.pyplot


        dataset['Gender'] = dataset['Gender'].map({0: 'Male', 1: 'Female'})

    elif selected_option == "Embarked Bar Plot":
        st.subheader("Embarked vs. Survival")
        embarked_survived = dataset.groupby('Embarked').Survived.value_counts(normalize=True).unstack(level=0)
        embarked_survived = embarked_survived * 100  # Convert to percentages
        fig_embarked_survival, ax = plt.subplots()
        embarked_survived.plot(kind='bar', ax=ax)
        embarked_survived.index = ['Not Survived', 'Survived']
        ax.set_xlabel('Embarked')
        ax.set_ylabel('Percentage')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')
        ax.set_xticklabels(['Not Survived', 'Survived'])
        ax.set_title('Survived Passengers by Embarked')
        st.pyplot(fig_embarked_survival)
        # Set the Seaborn color palette to Set2
        custom_palette = sns.color_palette("Set2")

        # Rest of your code remains the same
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x="Embarked", y="Survived", data=dataset, palette=custom_palette)

        ax.set_xticklabels(['Cherbourg', 'Queenstown','Southampton'])
        # Set axis labels and title
        plt.xlabel("Embarked")
        plt.ylabel("Survived (%)")
        plt.title("Survived Passengers by Embarked")

        # Format the y-axis labels as absolute percentages
        ax.set_yticklabels([f"{int(y*100)}%" for y in ax.get_yticks()])

        # Display the bar plot using st.pyplot
        st.pyplot(plt)
    elif selected_option == "Parch vs Survival":
        st.subheader("Parch vs. Survival")
        custom_palette=sns.color_palette("Set2") 
        # Create a bar plot using Seaborn
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        sns.barplot(x="Parch", y="Survived", ci=None, data=dataset, palette=custom_palette)
        # Set axis labels and title
        plt.xlabel("Parch(Number of parents of children travelling with each passenger.)")
        plt.ylabel("Survived")
        plt.title("Survived Passengers by Parch")

        # Display the bar plot using st.pyplot
        st.pyplot(plt)
    elif selected_option == "SibSp vs Survival":
        st.subheader("SibSp vs. Survival")
        sibsp_survived = dataset.groupby('SibSp').Survived.value_counts(normalize=True).unstack(level=0)
        sibsp_survived = sibsp_survived * 100  # Convert to percentages

        # Update the x-axis labels
        sibsp_survived.index = ['Not Survived', 'Survived']
        fig_sibsp_survival, ax = plt.subplots()
        sibsp_survived.plot(kind='bar', ax=ax)
        ax.set_xlabel('SibSp')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center')

        ax.set_ylabel('Percentage')
        ax.set_title('Survived Passengers by SibSp')
        st.pyplot(fig_sibsp_survival)


    elif selected_option == "Age vs. Survival":
        st.subheader("Age vs. Survival")
        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        # Create horizontal violin plots
        # sns.violinplot(y="Embarked", x="Age", hue="Survived", data=dataset, split=True, orient="h", ax=ax1)
        # sns.violinplot(y="Pclass", x="Age", hue="Survived", data=dataset, split=True, orient="h", ax=ax2)
        # sns.violinplot(y="Gender", x="Age", hue="Survived", data=dataset, split=True, orient="h", ax=ax3)
        sns.violinplot(y="Embarked", x="Age", hue="Survived", data=dataset, split=True, ax=ax1)
        sns.violinplot(y="Pclass", x="Age", hue="Survived", data=dataset, split=True, ax=ax2)
        sns.violinplot(y="Gender", x="Age", hue="Survived", data=dataset, split=True, ax=ax3)
        
        ax1.legend(title="Survival", labels=["Not Survived", "Survived"])
        ax2.legend(title="Survival", labels=["Not Survived", "Survived"])
        ax3.legend(title="Survival", labels=["Not Survived", "Survived"])
        st.pyplot(fig)
        # st.success("NOTE:")
        # st.success("1) From the Pclass violinplot, we can see that:")
        # st.success("- 1st Pclass has very few children as compared to other two classes.")
        # st.success("- 1st Plcass has more old people as compared to other two classes.")
        # st.success("- Almost all children (between age 0 to 10) of 2nd Pclass survived.")
        # st.success("- Most children of 3rd Pclass survived.")
        # st.success("- Younger people of 1st Pclass survived as compared to its older people.")

        # st.success("2) From the Sex violinplot, we can see that:")
        # st.success("- Most male children (between age 0 to 14) survived.")
        # st.success("- Females with age between 18 to 40 have a better survival chance.")
        st.markdown("""
    **NOTE:**
    1) From the Pclass violinplot, we can see that:

        - 1st Pclass has very few children as compared to other two classes.
        - 1st Plcass has more old people as compared to other two classes.
        - Almost all children (between age 0 to 10) of 2nd Pclass survived.
        - Most children of 3rd Pclass survived.
        - Younger people of 1st Pclass survived as compared to its older people.

    2) From the Sex violinplot, we can see that:

        - Most male children (between age 0 to 14) survived.
        - Females with age between 18 to 40 have a better survival chance.
    """)

        total_survived = dataset[dataset['Survived'] == 1]
        total_not_survived = dataset[dataset['Survived'] == 0]

        # Create a Streamlit figure
        fig = plt.figure(figsize=[15, 5])

        # Create a subplot
        plt.subplot(111)

        # Create distribution plots
        sns.distplot(total_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='blue')
        sns.distplot(total_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='red', axlabel='Age')

        # Set plot labels and titles

        # Display the plot using st.pyplot
        st.pyplot(fig)
        male_survived = dataset[(dataset['Survived'] == 1) & (dataset['Gender'] == 0)]
        female_survived = dataset[(dataset['Survived'] == 1) & (dataset['Gender'] == 1)]
        male_not_survived = dataset[(dataset['Survived'] == 0) & (dataset['Gender'] == 0)]
        female_not_survived = dataset[(dataset['Survived'] == 0) & (dataset['Gender'] == 1)]

        # Create a Streamlit figure
        fig = plt.figure(figsize=[15, 5])

        # Create the first subplot
        plt.subplot(121)
        sns.distplot(female_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='blue')
        sns.distplot(female_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='red', axlabel='Female Age')

        # Create the second subplot
        plt.subplot(122)
        sns.distplot(male_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='blue')
        sns.distplot(male_not_survived['Age'].dropna().values, bins=range(0, 81, 1), kde=True, color='red', axlabel='Male Age')

        # Display the plot using st.pyplot
        st.pyplot(fig)
        st.markdown("""
            **NOTE:**

            From the above figures, we can see that:
            - Combining both male and female, we can see that children with age between 0 to 5 have a better chance of survival.
            - Females with age between "18 to 40" and "50 and above" have a higher chance of survival.
            - Males with age between 0 to 14 have a better chance of survival.
            """)

        st.write("Question:")
        st.write("What observations can be made from the plots showing the relationship between Pclass, Age, Gender, and Survival?")

        options = [
            "1) 1st Pclass has more children compared to other two classes.",
            "2) 2nd Pclass has the highest survival rate among children aged 0 to 10.",
            "3) Most male children (age 0 to 14) have a higher survival rate.",
            "4) Women from 1st and 2nd Pclass have almost 100% survival chance.",
            "5) Younger people from 1st Pclass have a lower survival rate compared to older people."
        ]

        correct_option = "4) Women from 1st and 2nd Pclass have almost 100% survival chance."

        selected_option = st.radio("Choose the correct option:", options, index=None)
        if selected_option is not None:
            if selected_option == correct_option:
                st.success("That's correct! Women from 1st and 2nd Pclass indeed have almost 100% survival chance.")
            else:
                st.error("Sorry, that's not correct. Please try again.")



    else:
        st.write("Please select a valid option from the dropdown.")
elif action == "Casting the Net 🏂🏽":
    st.subheader('Casting the Net: Selecting Features for Prediction')
    st.markdown("""
    Cast your net wide to capture the most influential features for our predictive journey. Navigate the waters of feature selection, exploring the correlation currents and assessing the importance of each variable. We're strategically choosing the right features to ensure our predictive compass points true.
    """)
    sns.set_palette("Set2")
    st.subheader("Correlation with Survival")

    # Create a copy of the dataset
    temp_data = dataset.copy()

    # Map the 'Embarked' column using a dictionary
    embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
    temp_data['Embarked'] = temp_data['Embarked'].map(embarked_mapping)

    # Drop unnecessary columns
    temp_data = temp_data.drop(columns=['Name', 'PassengerId', 'Ticket', 'Cabin'])

    # Calculate the correlations with 'Survived'
    correlations = temp_data.corr()['Survived'].abs().drop('Survived')

    # Create a Streamlit figure
    fig, ax = plt.subplots(figsize=(15, 6))

    # Create a bar plot to visualize the correlations
    sns.barplot(x=correlations.index, y=correlations.values, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

    # Set labels and title
    ax.set_xlabel('Features')
    ax.set_ylabel('Correlation with Survived')
    ax.set_title('Correlation Between Features and Survived')
    st.pyplot(fig)
    st.info('By checking above Correlation with Survival, Could you test following Cases and their accuracy')
    st.info('CASE-I PClass,Gender and Fare')
    st.info('CASE-II PClass,Gender and Embark')
    st.info('CASE-II Age,Sibsp  and Parch')    
elif action == "Engines of Prediction: Constructing the Model ⚓":
    st.subheader("Feature selection")
    st.markdown("""
    Fire up the engines of prediction as we construct our machine learning model. Explore the various algorithms – decision trees, random forests, and logistic regression – each a different vessel on our analytical sea. Sail through the seas of model training, navigate the validation waters, and steer towards optimal performance.
    """)
    features_X = list(dataset.columns)

    cols_X = st.multiselect('Select the input features(X)', features_X)
    st.info("Output features 'Survived' is selected.")
    # Fix the output feature to 'Survived'
    cols_Y = ['Survived']

    X = dataset[cols_X]
    y = dataset[cols_Y]

    st.write("Input features: ", X.head())
    st.write("Output features: ", y.head())

    # missing data

    # st.subheader("Handling missing data")


    # Filter numeric columns (int or float) for handling missing data
    numeric_cols = X.select_dtypes(include=['number']).columns
    # missing_cols = st.multiselect('Select the features', numeric_cols)
    missing_cols= numeric_cols
    if len(missing_cols) and len(X):
        missing_strategy = 'mean'

        execute_title_extraction = False

        try:
            if missing_strategy == 'mean':
                si = SimpleImputer(strategy='mean')
                X[missing_cols] = si.fit_transform(X[missing_cols])
            elif missing_strategy == 'median':
                si = SimpleImputer(strategy='median')
                X[missing_cols] = si.fit_transform(X[missing_cols])
            elif missing_strategy == 'most_frequent':
                si = SimpleImputer(strategy='most_frequent')
                X[missing_cols] = si.fit_transform(X[missing_cols])
            elif missing_strategy == 'constant':
                c = st.text_input("Constant: ")
                si = SimpleImputer(strategy='constant', fill_value=c)
                X[missing_cols] = si.fit_transform(X[missing_cols])
            elif missing_strategy == 'drop_row':
                X.dropna(subset=missing_cols, inplace=True)
                X.reset_index(drop=True, inplace=True)

            if 'Name' in cols_X and execute_title_extraction:
                for dataset in train_test_data:
                    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.')
                    st.write("Title column: ", dataset['Title'])  # Print the Title column
        except Exception as e:
            st.error("An error occurred: " + str(e))
    else:
         st.warning("No Feature column is selected")
        # st.write("Input features without handling missing data: ", X)

    if len(missing_cols) and len(X):
        temp_data_for_split=X.copy()
        y_dummy=y.copy()
        st.subheader("Splitting the data into training and test sets")
        categorical_cols = X.select_dtypes(include=['object']).columns

        # Apply one-hot encoding to these categorical columns
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        try:
            train_test_ratio = st.slider('Select the test set ratio', 0.1, 0.3, step=0.01)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_test_ratio, random_state=0)
            temp_data_for_split_train,temp_data_for_split_test,y_dummy_train,y_dummy_test=train_test_split(temp_data_for_split, y_dummy, test_size=train_test_ratio, random_state=0)
            st.write('Rows in the train set =', len(y_dummy_train))
            st.write('Rows in the test set =', len(y_dummy_test))
        except Exception as e:
            print(e)
            st.error("There is some error")

        st.write("Training set: ", temp_data_for_split_train.head(), y_dummy_train.head())
        st.write("Testing set: ", temp_data_for_split_test.head(), y_dummy_test.head())
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        st.header("Constructing the AI Model")
        st.info("Decision Tree algorithm model is used to build the model")
        algos = ['Logistic Regression', 'K-NN', 'SVM', 'Naive Bayes', 'Decision tree', 'Random forest']
        algo_selected = 'Decision tree'
        # Sidebar checkboxes

        criterion_ ='gini'
        max_ = 'log2'
        
        decision_tree_model = DecisionTreeClassifier(criterion=criterion_, random_state=0, max_features=max_)

        decision_tree_model.fit(X_train, y_train)
        y_pred = decision_tree_model.predict(X_test)
        decision_tree_pred = decision_tree_model.predict(X_test)
        # Fit the model
        
        decision_tree_model.fit(X_train, y_train)

        st.write(confusion_matrix(y_test, decision_tree_pred))
        acc = accuracy_score(y_test, decision_tree_pred)

        # Display accuracy
        if acc > 0.7:
            st.success('Accuracy = {:.4f}%'.format(acc * 100))
        elif acc > 0.6:
            st.info('Accuracy = {:.4f}%'.format(acc * 100))
        elif acc > 0.5:
            st.warning('Accuracy = {:.4f}%'.format(acc * 100))
        else:
            st.error('Accuracy = {:.4f}%'.format(acc * 100))


            # Section for testing the model on custom features
elif action =="Safe Harbors: Concluding the Predictive Odyssey 🌊":
    st.subheader('Safe Harbors: Concluding your model 🌊')
    st.markdown("""
    As our predictive odyssey concludes, reflect on the discoveries made. Summarize the tales told by the data, the impact of chosen features, and the prowess of our machine learning model. Our journey ends, leaving us with a newfound understanding of the Titanic disaster and the power of prediction.
    """)
    categorical_cols = X_final.select_dtypes(include=['object']).columns
    custom_data_list = []
    # Apply one-hot encoding to these categorical columns
    X_final = pd.get_dummies(X_final, columns=categorical_cols, drop_first=True)
    criterion_ ='gini'
    max_ = 'log2'
    decision_tree_model = DecisionTreeClassifier(criterion=criterion_, random_state=0, max_features=max_)

    decision_tree_model.fit(X_final, y_final)

    with st.form(key='feature_form'):
        gender_mapping = {"Male": 0, "Female": 1}
        custom_feature_sliders = {}
        selected_gender = st.radio(f"Select Categorical Attribute for Gender", ["Male", "Female"])
        custom_feature_sliders['Gender'] = gender_mapping[selected_gender]
        custom_data_list.append(custom_feature_sliders['Gender'])

        pclass_mapping = {"First Class": 1, "Second Class": 2, "Third Class": 3}
        selected_pclass = st.radio(f"Select Categorical Attribute for Pclass", ["First Class", "Second Class", "Third Class"])
        custom_feature_sliders['Pclass'] = pclass_mapping[selected_pclass]
        custom_data_list.append(custom_feature_sliders['Pclass'])
        embarked_mapping = {"Cherbourg": "C", "Queenstown": "Q", "Southampton": "S"}
        selected_embarked = st.radio(f"Select Categorical Attribute for Embarked", ["Cherbourg", "Queenstown", "Southampton"])
        
        
        custom_feature_sliders['Embarked'] = embarked_mapping[selected_embarked]
        custom_data_list.append(custom_feature_sliders['Embarked'])
        custom_feature_sliders['Name'] = st.text_input(f"Enter a value for Name")

        # Use the model to make predictions on custom data
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:

        # Apply one-hot encoding to categorical features
        custom_data = pd.DataFrame(index=[0], columns=X_final)
        # Apply one-hot encoding to categorical features
        categorical_cols = custom_data.select_dtypes(include=['object']).columns
        custom_data_encoded = pd.get_dummies(custom_data, columns=categorical_cols, drop_first=True)
        missing_cols = set(X_final.columns) - set(custom_data_encoded.columns)
        for col in missing_cols:
            custom_data_encoded[col] = 0

        # Reorder columns to match training data
        custom_data_encoded = custom_data_encoded[X_final.columns]

        # Use the model to make predictions on custom data
        custom_predictions = decision_tree_model.predict(custom_data_encoded)

        # Code to execute when the form is submitted
        st.subheader("Model Predictions on Custom Features:")
        st.write("Predicted Survival:", "Survived" if custom_predictions[0] == 1 else "Not Survived")
        if custom_predictions[0] == 1:
            st.image("survived.jpeg", caption="Survived", use_column_width=True)
        else:
            st.image("not_survived.jpg", caption="Not Survived", use_column_width=True)

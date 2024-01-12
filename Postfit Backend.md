## Project Structure

The project is divided into separate _apps_, each encapsulating its functionality:

```
<YourProjectName>/
│
├── apps/
│   ├── <YourAppName>/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── services.py
│   │   └── ...
│   ├── ...
│
├── shared/
│   ├── __init__.py
│   ├── database.py
│   └── log_config.py
│
├── config.py
├── main.py
├── README.md
└── requirements.txt
```

### `/apps`

Each app within the `apps/` directory contains:

- `__init__.py`: Initializes a Flask Blueprint and registers API resources.
- `models.py`: Defines SQLAlchemy models related to the app.
- `views.py`: Contains Flask-RESTful resources that define the endpoints for the app.
- `services.py`: Holds business logic and interactions with the database.
- `schemas.py`: (If present) Contains Marshmallow schemas for serialization and deserialization.

### `/instance`

This directory holds the `development.db` sqlite database for testing and development

### `/logs`

This directory stores the log files of each app module

### `/shared`

The `shared/` directory contains components that are common across different apps:

- `database.py`: Sets up the SQLAlchemy database instance.
- `log_config.py`: Configures the application logging.

### `seed.py`

Simple seed. Run the following in terminal from root:

```
python seed.py
```

This will clear out the `database.db` and refill it with initial dummy data for development purposes

### `config.py`

This file holds configuration details. It serves as a macro to switch from development to production environments

### `/open_ai`

This module holds functionality that has to do with interacting with [OpenAI's API](https://platform.openai.com/). The module contains two files `gpt.py` and `prompts.py`

**`gpt.py`** contains the following methods

- `ask_gpt`: responsible for sending prompts to the OpenAI API and retrieving responses
- `generate_workout_plan` helper function which takes user data (including parameters like age, gender, Physical Activity Level) and generates a custom prompt for a workout plan

**`prompts.py`** contains an abstract class `Prompt` from which all other prompt classes inherit from. It serves as a base class for different types of prompts, enforcing a standard structure with methods like `generate_request`, `format_response`, and `validate_response_structure`. The abstract methods ensure that any subclass must provide specific implementations for generating and handling prompts.

---

## Getting Started

### Dependencies

- Python 3.x  
- Flask 
- Flask-RESTful
- Flask-Migrate 
- Flask-SQLAlchemy 
- Marshmallow 
- OpenAI
- Typing

*(see requirements.txt for full list)*

### To get started, follow these steps:

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Initialize the database:

```
flask db upgrade
```

3. Run the application:

```
flask run
```

or

```
python app.py
```


this will run the application on port 5000

### Creating a New App

To create a new app within the project, run the `startapp.py` script with the name of the app:

```bash
python startapp.py <AppName>
```

This will create a new directory under `apps/` with the appropriate file structure.

### Logging

Each app configures its own logger in `logger.py`, which logs to both the console and a file within the `logs/` directory. The `overwrite` argument is set to `TRUE` by default, which means the logs will be overwritten each active session. To change this, simply modify the argument to `FALSE`

---

## Business Logic: Implementation Detail of Current Apps

### Diet App

The Diet app is dedicated to managing all diet-related functionalities. It's structured to encapsulate models, views, schemas, and services specific to dietary management, ensuring that concerns are cleanly separated in line with our modular application design.

#### Models

All diet related models should go in this app. In its current stage, the sole model in the Diet app is the `MealEntry`, representing user-submitted dietary data generated from the data retrieved by the [Passio API](https://www.passio.ai/nutrition-ai?utm_term=nutrition%20api&utm_campaign=Nutrition-AI+Page&utm_source=adwords&utm_medium=ppc&hsa_acc=6097232191&hsa_cam=19981831113&hsa_grp=149225562858&hsa_ad=655216717362&hsa_src=g&hsa_tgt=kwd-2009999184384&hsa_kw=nutrition%20api&hsa_mt=b&hsa_net=adwords&hsa_ver=3&gad_source=1&gclid=Cj0KCQiAkeSsBhDUARIsAK3tiefjKn7FDZlJAesj3QlcTzJtL_jIyyeQ_pMYUuT3jz4Jvemas2Cjo7MaAnkwEALw_wcB). Each `MealEntry` contains nutritional information about food items, aggregated into a comprehensive nutritional profile for the user.

Here's a brief overview of the `MealEntry` model:

```python
class MealEntry(db.Model):
	__tablename__ = 'meal_journal_entries'
	id = db.Column(db.Integer, primary_key=True)
	member_id = db.Column(db.Integer, db.ForeignKey('members.memberID'), nullable=False)
	# 'entry' column will store a list of food items as a JSON structure
	entry = db.Column(db.JSON, nullable=False)
	# 'totals' column will store the aggregated nutrition totals as a JSON structure
	totals = db.Column(db.JSON, nullable=False)
	created_at = db.Column(db.DateTime, default=datetime.utcnow)
	updated_at = db.Column(
	db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	def __repr__(self):
		return f'<MealEntry {self.id}>'
```

#### Views

The views in the Diet app provide RESTful API endpoints to create, retrieve, update, and delete meal entries. These views are defined using Flask-RESTful's `Resource` class for clean and maintainable API design.

Here's an example of a view from the Diet app:

```python
class MealEntry(Resource):
    # ... HTTP methods implementations ...
```

#### Services

The service layer, exemplified by the `DietService` class, contains the business logic for the Diet app. This layer interacts with the database and performs operations such as calculating nutritional totals for meal entries.

Here's an example method from the `DietService` class:

```python
class DietService:
    @staticmethod
    def create_meal_entry(member_id, entry):
        # ... logic to create a meal entry ...
```

#### Endpoints

The Diet app exposes the following RESTful endpoints:

- `GET /meal-entries/<int:meal_entry_id>`: Retrieve a specific meal entry.
- `PUT /meal-entries/<int:meal_entry_id>`: Update a specific meal entry.
- `DELETE /meal-entries/<int:meal_entry_id>`: Delete a specific meal entry.
- `GET /meal-entries/`: Retrieve a list of meal entries within a range.
- `POST /meal-entries/`: Create a new meal entry.

#### Usage

To use the Diet app, start by creating meal entries via the provided API endpoints. Each entry should be accompanied by the user's nutritional data obtained from the Passio nutrition API.

> The entry field is a list where each item represents a distinct food item consumed in a meal _(I/e: a meal of fish and chips would have an entry that is a list of length 2)_. Each food item will contain the following attributes taken directly from the [PassioFoodItemData](https://pub.dev/documentation/nutrition_ai/latest/nutrition_ai/PassioFoodItemData-class.html) class:

- **barcode**: A unique identifier for the food item.
- **entityType**: The type of the item, e.g., 'food'.
- **ingredientsDescription**: A description of the food item, e.g., 'Beef steak, grilled'.
- **name**: The name of the food item, e.g., 'Grilled Steak'.
- **passioID**: A unique identifier specific to the application's database.
- **selectedQuantity**: The quantity of the food item consumed.
- **selectedUnit**: The unit of measurement for the quantity, e.g., 'grams'.
- **servingSize**: The standard serving size of the food item.
- **servingUnits**: The unit of measurement for the serving size.
- **calories**: The caloric content of the food item.
- **calcium**: The amount of calcium in the food item.
- **carbs**: The carbohydrate content of the food item.
- **fat**: The fat content of the food item.
- **proteins**: The protein content of the food item.
- **cholesterol**: The cholesterol content of the food item.

> POST request payload from client

```json
{
	"entry": [
		{
			"barcode": "987654320",
			"entity_type": "food",
			"ingredientsDescription": "Mixed fruit salad",
			"name": "Fruit Salad",
			"passio_id": "fruitsalad1",
			"selected_quantity": 1,
			"selected_unit": "grams",
			"serving_size": 50,
			"serving_units": "grams",
			"calories": 150,
			"calcium": 20,
			"carbs": 35,
			"fat": 1,
			"proteins": 2,
			"cholesterol": 0
		},
		{
			"barcode": "111222333",
			"entity_type": "food",
			"ingredients_description": "Yogurt, low-fat",
			"name": "Low-fat Yogurt",
			"passio_id": "yogurt1",
			"selected_quantity": 1,
			"selected_unit": "grams",
			"serving_size": 50,
			"serving_units": "grams",
			"calories": 60,
			"calcium": 150,
			"carbs": 10,
			"fat": 2,
			"proteins": 5,
			"cholesterol": 10
		}
	]
}
```

> Instance Persisted to Database

```json
{
	"id": 5,
	"entry": [
		{
			"barcode": "987654320",
			"entity_type": "food",
			"ingredients_description": "Mixed fruit salad",
			"name": "Fruit Salad",
			"passio_id": "fruitsalad1",
			"selected_quantity": 1.0,
			"selected_unit": "grams",
			"serving_size": 50.0,
			"serving_units": "grams",
			"calories": 150.0,
			"calcium": 20.0,
			"carbs": 35.0,
			"fat": 1.0,
			"proteins": 2.0,
			"cholesterol": 0.0
		},
		{
			"barcode": "111222333",
			"entityType": "food",
			"ingredients_description": "Yogurt, low-fat",
			"name": "Low-fat Yogurt",
			"passio_id": "yogurt1",
			"selected_quantity": 1.0,
			"selected_unit": "grams",
			"serving_size": 50.0,
			"serving_units": "grams",
			"calories": 60.0,
			"calcium": 150.0,
			"carbs": 10.0,
			"fat": 2.0,
			"proteins": 5.0,
			"cholesterol": 10.0
		}
	],
	"totals": {
		"totalCalories": 210.0,
		"totalCalcium": 170.0,
		"totalCarbs": 45.0,
		"totalFat": 3.0,
		"totalProteins": 7.0,
		"totalCholesterol": 10.0
	},
	"created_at": "2024-01-05T00:29:10.915433",
	"updated_at": "2024-01-05T00:29:10.915438"
}
```

### Fitness App

All fitness related functionality will be included here. One of the key functions is to generate and manage personalized workout plans. Leveraging the capabilities of Chat GPT, the app creates a 7-day workout regimen tailored to the user's specific goals and physical activity level.

#### Model

At the heart of the Fitness app is the `WorkoutPlan` model, designed to store individual workout plans:

```python
class WorkoutPlan(db.Model):
    __tablename__ = 'workout_plans'
    # ... fields and methods ...
```

#### Key Features

- **Personalization**: Each workout plan is crafted based on user-provided data, ensuring that it aligns with their fitness objectives.
- **Integration with Chat GPT**: Utilizes the `open_ai` module to interact with Chat GPT, which processes user data to generate a customized workout plan.

#### RESTful Endpoints

The app offers a suite of RESTful endpoints that facilitate the creation, retrieval, and management of workout plans:

- `GET /workout-plans/<int:workout_plan_id>`: Fetches a specific workout plan by its ID.
- `POST /workout-plans/`: Initiates the generation of a new workout plan using data provided in the request payload.
- `PUT /workout-plans/<int:workout_plan_id>`: Allows for the updating of workout plans, including modifications to the generated suggestions.
- `DELETE /workout-plans/<int:workout_plan_id>`: Removes a workout plan from the database.

#### Usage

To generate a workout plan, users submit their physical and dietary data through the endpoint, which is then processed to create a workout plan that not only matches their fitness goals but also accommodates their dietary needs and restrictions.

#### Services

The `WorkoutService` class provides the necessary business logic, including:

- `calculate_PAL`: Computes the Average Physical Activity Level (PAL) based on user data.
- `process_workout_plan`: Orchestrates the workout plan generation process.
- CRUD operations: Functions to create, retrieve, update, and delete workout plans.

```python
class WorkoutService:
    # ... service methods ...
```

#### Usage

To start using the Fitness app, simply send a POST request to `/workout-plans/` with the required data, and the service will handle the rest, from interacting with Chat GPT to persisting the workout plan.

> POST request payload from client

```json
{
	"prompt_type": "workout_plan",
	"weight_loss_goal": 3,
	"weight_loss_duration": 30,
	"caloric_intake": 2500,
	"weight": 81.65,
	"height": 180,
	"age": 28,
	"gender": "male",
	"workout_duration": 90
}
```

> Instance Persisted to Database

```json
{
	"id": 1,
	"start_date": "2024-01-04T23:56:53.540623",
	"gpt_suggestion": [
		"Day 1: Cardiovascular - 30 minutes of jogging or running, Strength - 3 sets of 10 reps of push-ups, squats, and lunges, Flexibility - 15 minutes of stretching and yoga",
		"Day 2: Rest day",
		"Day 3: Cardiovascular - 30 minutes of swimming, Strength - 3 sets of 10 reps of dumbbell chest press, deadlifts, and bicep curls, Flexibility - 15 minutes of stretching and yoga",
		"Day 4: Cardiovascular - 30 minutes of cycling or spinning, Strength - 3 sets of 10 reps of overhead press, bent-over rows, and tricep dips, Flexibility - 15 minutes of stretching and yoga",
		"Day 5: Rest day",
		"Day 6: Cardiovascular - 30 minutes of HIIT workouts, alternating between sprinting and bodyweight exercises, Strength - 3 sets of 12 reps of Bulgarian split squats, dumbbell chest flys, and pull-ups, Flexibility - 15 minutes of stretching and yoga",
		"Day 7: Cardiovascular - 30 minutes of jump rope, Strength - 3 sets of 12 reps of kettlebell swings, bench press, and lateral raises, Flexibility - 15 minutes of stretching and yoga"
	]
}
```
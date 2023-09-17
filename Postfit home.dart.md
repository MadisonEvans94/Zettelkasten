
## Diagram

![[Untitled Page 2.pdf]]

## Functions

### `initState()`

---
### `checkUserAccountStatus()`

---


### `reactivateYourAccount()`

---


### `didChangeDependencies()`

---


### `startTimer()`

---

### `addMinutes()`

```dart
DateTime addMinutes(DateTime dateTime) {
    ...
}
```

**Description**: A utility function that adds 30 minutes to a given `DateTime` object and then returns the new `DateTime` value.

**Parameters**:

- `dateTime`: The original DateTime value to which 30 minutes will be added.

---

### `getUserDiet()`

- Fetches an authentication token.
- Uses an API service (likely provided through the `Provider` package) to check the user's diet.
- Decodes the API response.
- If the user has saved diet information, it updates the state and the `ChangeDietPlan` provider with that information. Otherwise, it tells the `ChangeDietPlan` provider to delete the plan.

---

### `getBookingList()`

- Constructs a date and timezone-related information.
- Fetches an authentication token.
- Uses an API service to get the current booking.
- Parses the API response and processes the data:
    - It updates various states in the widget based on the API's response, such as the user's journey, subscription status, remaining days, free trial data, etc.
    - Processes upcoming events, sorting them, and adding a countdown for each.
    - Checks for live sessions and adds them to the booking list.
    - Adds challenges to a challenge list.

---


### `generatePromoCode()`

---

## Widgets 

### `_dividerWidget()`

```dart
Widget _dividerWidget() {
    return Divider(
      height: 10,
      thickness: 1,
      color: Colors.black,
    );
}
```

**Description**: This widget returns a horizontal divider with a height of `10`, a thickness of `1`, and a black color.

---

### `_profilePicture()`

```dart
Widget _profilePicture() {
    ...
}
```

**Description**: This widget displays the user's profile picture if it exists, otherwise it shows a default image. The user's profile picture is fetched using the `FadeInImage.assetNetwork` widget which helps to smoothly show an image by displaying a placeholder image until the target image is loaded.

**Dependencies**: 

- AppData model for fetching the user's profile picture URL.
- Images class to provide the path to the placeholder image.

---

### `_topMenuHeader()`

```dart
Widget _topMenuHeader(String optionText, int postion) {
    ...
}
```

**Description**: This widget displays a circle avatar containing an image which corresponds to the given `position`. It also has an `onTap` behavior defined based on the `position` value. Additionally, it displays text below the avatar.

**Parameters**:

- `optionText`: The text to be displayed below the circle avatar.
- `position`: Determines the image and behavior of the avatar. Valid values:
  - `1`: Tracker image and behavior.
  - `2`: Diet Plan image and behavior.
  - `3`: Live Fitness image and behavior.
  - `4`: Free Trial image and behavior.

**Dependencies**: 

- `AppImages` class for fetching image paths.
- `ChangeDietPlan` model for checking diet plan status.
- Navigation to different pages is determined by the given `position` value.

---
### `buildScheduleBookingTimer()`

```dart
Widget buildScheduleBookingTimer(BookingData bookingData) {
    ...
}
```

**Description**: This widget displays a countdown timer with hours, minutes, and seconds arranged horizontally. It also provides a "Join" button at the bottom.

**Parameters**:

- `bookingData`: Data associated with the booking, though it seems unused in the provided context.

**Dependencies**:

- Utility function `twoDigits` for formatting time.
- `duration` seems to be an external variable or property (its initialization is not provided in the code snippet).
- `SizeConfig` for scaling the dimensions.
- `AppColors` for color definitions.

---
### `colonIcon()`

```dart
Widget colonIcon() {
    ...
}
```

**Description**: This widget displays two small rounded circle icons vertically, representing a colon, typically used in between hours, minutes, and seconds in time displays.

**Dependencies**: 

- `AppColors` for color definitions.

---
### `buildTimeCard()`

```dart
Widget buildTimeCard({required String time, required String header}) {
    ...
}
```

**Description**: This widget showcases a card that displays a time value (like hours, minutes, or seconds) along with its header (like "Hr", "Min", "Sec"). 

**Parameters**:

- `time`: The time value to be displayed.
- `header`: The header or title for the time value (e.g. "Hr" for hours).

**Dependencies**: 

- `SizeConfig` for scaling dimensions.
- `AppColors` for color definitions.

---

### `addStopDownCountDown()`

```dart
Widget addStopDownCountDown(String type, int i) {
    ...
}
```

**Description**: This widget displays a countdown timer with relevant textual information, based on the booking data and type. At the bottom of this widget, there's an actionable button which can either say "View" or "View All" based on conditions. Tapping on this button will either navigate to the `LiveFitnessScreen`, `ScheduleMeetingScreen`, or prompt the user to subscribe based on certain conditions.

**Parameters**:

- `type`: Determines whether the booking is a "meeting" or something else.
- `i`: Index to fetch data from the `bookingList`.

**Dependencies**:

- `MediaQuery` for screen dimension adjustments.
- `CountDownTimer` widget for showing the countdown.
- `CustomButton` for the action button.
- `AppColors` for color definitions.

**Actions**:

1. On tapping the `GestureDetector`:
	- Navigates to the `LiveFitnessScreen` if the trainer's name is "Live session" and the user is subscribed.
	- Shows a dialog prompting the user to subscribe if the trainer's name is "Live session" and the user isn't subscribed.
	- Navigates to the `ScheduleMeetingScreen` for other cases.

---


### `dialogContentCongratulations()`

```dart
Widget dialogContentCongratulations(BuildContext context) {
    ...
}
```

**Description**: A widget that returns a dialog content to congratulate the user on unlocking a free trial of VIP Membership. The content details the features of this membership.

**Dependencies**:

- `SizeConfig` for scaling dimensions.
- `SubscriptionChangePlan` for fetching and displaying the remaining free trial days.
- `MyTheme` for styling and decoration.

**Actions**:

1. On pressing the "Close" button:
    - Closes the current dialog.
    - Navigates to the `GetStartedScreen`.
    - If the return value from the `GetStartedScreen` is true, shows the `showGenerateCouponDialog`.

---

### `CarouselController()`

```dart
CarouselController() {
	...
}
```

---

### `showGenerateCouponDialog()`

```dart
showGenerateCouponDialog() {
	...
}
```

---
### `dialogContentPromoCode()`

```dart
dialogContentPromoCode(BuildContext context, String promote) {
	...
}
```

---

## Alerts

### `ShowActivateAccountDialog()`

```dart
ShowActivateAccountDialog() {
	...
}
```

---
### `_showMyDialog()`

```dart
Future<void> _showMyDialog(bool subscribe, int days) async {
    ...
}
```

**Description**: A utility function that displays an alert dialog which informs the user about VIP Membership and its status (whether they need to subscribe or the remaining days of their trial). The dialog provides an option for the user to either subscribe immediately or simply close the dialog.

**Parameters**:

- `subscribe`: A boolean to determine the message to be displayed. If `true`, the user is prompted to subscribe. Otherwise, it displays the remaining days of the trial.
- `days`: The number of days left in the trial. This is used when `subscribe` is `false`.

**Dependencies**:

- `SizeConfig` for scaling dimensions.
- `MyColor` for color definitions.

**Actions**:

1. On pressing "Subscribe Now" button:
	- Closes the current dialog.
	- Navigates to the `MembershipScreen`.
2. On pressing "Close" button:
	- Simply closes the dialog.

---
### `openRedeemFreeSession()`

```dart
openRedeemFreeSession(bool value) async {
    ...
}
```

**Description**: Opens the congratulations dialog if the provided value is true.

**Parameters**:

- `value`: A boolean value which, if true, triggers the display of the congratulations dialog.

**Actions**:

1. If `value` is true, shows the congratulations dialog with non-dismissible behavior.

---

### `checkConnection()`

```dart
Future<bool> checkConnection() async {
    ...
}
```

**Description**: Checks if there is an active internet connection (either via mobile data or Wi-Fi) and returns a boolean indicating the result.

**Dependencies**:

- `Connectivity` package to check the type of connection.

**Returns**:

- `true` if there's an active internet connection (either mobile or Wi-Fi).
- `false` otherwise.

---

### `showNoInternereDialog()`

```dart
showNoInternereDialog() {
    ...
}
```

**Description**: Displays an alert dialog to inform the user that there's no active internet connection and provides a "retry" option.

**Dependencies**:

- `AppStrings` and `AppColors` for string constants and color values respectively.
- `checkConnection` function to recheck the internet connection.
- A method `checkUserAccountStatus` (not provided in the snippet) that seems to be called once an internet connection is established.

**Actions**:

1. On pressing the "Retry" button:
    - The dialog is closed.
    - The connection status is rechecked.
    - If there's still no internet connection, the dialog is redisplayed.
    - If there's an internet connection, the `checkUserAccountStatus` function is called.

---
#evergreen1 
upstream: [[Web API]]

---

**video links**: 

---

Sure, here's a simple guide on how to use Postman to test the Express application you're running locally.

---

## Testing an Express Application with Postman

### 1. Start the Express Server

In your terminal or command prompt, navigate to the directory where your Express application resides. Start your server by running `npm start` or `nodemon`, whichever command you've set up to run your server.

### 2. Open Postman

Open the Postman app on your computer. 
![[Screen Shot 2023-06-25 at 11.04.12 AM.png]]

### 3. Create a New Request

You should see a button that says **New**. Click on it 

![[Screen Shot 2023-06-25 at 11.04.57 AM.png]]

and select the type of request you want to test with from the menu.
![[Screen Shot 2023-06-25 at 11.08.34 AM.png]]
### 4. Test the GET Endpoints

#### Test the root endpoint:

1. In the "Enter request URL" box, enter `http://localhost:5001/` or whatever port your Express app is running on.
2. Make sure the HTTP method to the left of this box is set to `GET`.
3. Click the "Send" button.
4. You should see the message "Welcome to the app!" displayed in the "Body" tab of the "Response" section below (or whatever resource you have at your endpoint)

#### Test the `/api/` endpoint:

1. Change the request URL to `http://localhost:5001/api/`.
2. Click the "Send" button.
3. You should see the message "Welcome to the api" displayed in the "Body" tab of the "Response" section.

#### Test the `/api/student` endpoint:

1. Change the request URL to `http://localhost:5001/api/student`.
2. Click the "Send" button.
3. You should see a list of all students in the "Body" tab of the "Response" section. If no students exist yet, this should be an empty array.

#### Test the `/api/student/:id` endpoint:

1. Add an ID to the request URL, e.g., `http://localhost:5001/api/student/1`.
2. Click the "Send" button.
3. You should see the data of the student with this ID in the "Body" tab of the "Response" section.

### 5. Test the POST Endpoints

#### Test the `/api/student` endpoint:

1. Set the HTTP method to `POST`.
2. Set the request URL to `http://localhost:5001/api/student`.
3. Click on the "Body" tab below the URL field.
4. Select the "raw" option and then select "JSON" from the dropdown menu that appears to the right.
5. In the text field that appears, enter the new student's data in JSON format, for example:

    ```json
    {
      "name": "John Doe"
    }
    ```

6. Click the "Send" button.
7. In the "Body" tab of the "Response" section, you should see the data for the student you just added, including an automatically assigned ID.

### 6. Test Other Endpoints

Continue in this manner to test any other endpoints your Express app provides.

---

That's it! This guide should cover the basic steps to test your Express application with Postman. Remember, replace `5001` with the port your server is running on if it's different. Postman is a powerful tool and has many more features you may find useful, including saving requests for future use, running collections of requests, and more.



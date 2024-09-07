# 🎉 Welcome to Emotion Sense API! 

This API is built using **FastAPI** and is designed to perform **sentiment analysis** on text inputs. Whether it's identifying emotions like positive, negative, or neutral, this API has got you covered!

🌐 **Live Demo**: The API is deployed on **Azure Web Services** and can be accessed at the following link:  
[Emotion Sense API](https://emotionsense-dvemggcjagcqdxey.uaenorth-01.azurewebsites.net/)

## POST /predict

This endpoint takes in a text input and returns a sentiment prediction. The sentiment is classified as `positive`, `neutral`, or `negative` based on the analysis of the text.

### Request:

- **URL**: `/predict`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Request Body**:
  - The request body should contain a string of text to be analyzed for sentiment.

### Response:

- **Content-Type**: `application/json`
- **Body**: A JSON object containing a `prediction` field with an integer value:
  - `0` = neutral
  - `1` = positive
  - `2` = negative

#### Example response:
```json
{
    "prediction": 1
}

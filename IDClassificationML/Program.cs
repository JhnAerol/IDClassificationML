//using IDClassificationML; 
//using OpenCvSharp;
//using System;
//using System.IO;

//class Program
//{
//    static void Main(string[] args)
//    {
//        using var capture = new VideoCapture(0); 
//        using var window = new Window("Live Camera");

//        if (!capture.IsOpened())
//        {
//            Console.WriteLine("Camera not detected!");
//            return;
//        }

//        var frame = new Mat();

//        while (true)
//        {
//            capture.Read(frame);
//            if (frame.Empty()) continue;

//            window.ShowImage(frame);

//            byte[] imageBytes = frame.ToBytes(".jpg");

//            var input = new MLModel.ModelInput()
//            {
//                ImageSource = imageBytes
//            };

//            var result = MLModel.Predict(input);

//            float predictedScore = 0f;
//            int labelIndex = Array.IndexOf(result.Score, result.Score.Max());
//            if (labelIndex >= 0)
//                predictedScore = result.Score[labelIndex];

//            Console.WriteLine($"Predicted Label: {result.PredictedLabel}, Score: {predictedScore:P2}");

//            if (result.PredictedLabel == "ValidID" && predictedScore >= 0.9995f)
//            {
//                Console.WriteLine("Valid ID detected with high confidence, stopping...");
//                break;
//            }

//            if (Cv2.WaitKey(1) == 27) break;
//        }
//    }
//}

using IDClassificationML;
using OpenCvSharp;
using System;
using System.IO;
using Tesseract;
using System.Text.RegularExpressions;

class Program
{
    static void Main(string[] args)
    {
        using var capture = new VideoCapture("http://10.0.1.203:8080/video");
        using var window = new Window("Live Camera");

        if (!capture.IsOpened())
        {
            Console.WriteLine("Camera not detected!");
            return;
        }

        var frame = new Mat();

        string tessDataPath = Path.Combine(Directory.GetCurrentDirectory(), "tessdata");
        if (!Directory.Exists(tessDataPath))
        {
            Console.WriteLine("Error: tessdata folder not found!");
            return;
        }

        while (true)
        {
            capture.Read(frame);
            if (frame.Empty()) continue;

            window.ShowImage(frame);

            byte[] imageBytes = frame.ToBytes(".jpg");

            var input = new MLModel.ModelInput()
            {
                ImageSource = imageBytes
            };

            var result = MLModel.Predict(input);

            float predictedScore = 0f;
            int labelIndex = Array.IndexOf(result.Score, result.Score.Max());
            if (labelIndex >= 0)
                predictedScore = result.Score[labelIndex];

            Console.WriteLine($"Predicted Label: {result.PredictedLabel}, Score: {predictedScore:P2}");

            if (result.PredictedLabel == "ValidID" && predictedScore >= 0.9995f)
            {
                Console.WriteLine("Valid ID detected with high confidence, extracting text...");

                
                int mm3_in_pixels = 16;

                int nameWidth = frame.Cols / 2; 
                int nameHeight = frame.Rows / 10;
                int nameX = (frame.Cols - nameWidth) / 2; 
                int nameY = mm3_in_pixels; 

                var nameRect = new OpenCvSharp.Rect(nameX, nameY, nameWidth, nameHeight);

                int idWidth = frame.Cols / 2;
                int idHeight = frame.Rows / 10;
                int idX = (frame.Cols - idWidth) / 2;
                int idY = (frame.Rows - idHeight) / 2;

                var idRect = new OpenCvSharp.Rect(idX, idY, idWidth, idHeight);

                Mat nameCrop = new Mat(frame, nameRect);
                Mat idCrop = new Mat(frame, idRect);

                string tempNameFile = Path.Combine(Path.GetTempPath(), "tempName.jpg");
                string tempIdFile = Path.Combine(Path.GetTempPath(), "tempId.jpg");
                Cv2.ImWrite(tempNameFile, nameCrop);
                Cv2.ImWrite(tempIdFile, idCrop);

                using var ocr = new TesseractEngine(tessDataPath, "eng", EngineMode.Default);

                // OCR for Name
                using var nameImg = Pix.LoadFromFile(tempNameFile);
                using (var namePage = ocr.Process(nameImg)) {
                    string nameText = namePage.GetText();
                    var nameMatch = Regex.Match(nameText, @"Name[:\s]*(.+)");
                    if (nameMatch.Success)
                        Console.WriteLine($"Name: {nameMatch.Groups[1].Value.Trim()}");
                    else
                        Console.WriteLine("Name not found!");
                }

                using var idImg = Pix.LoadFromFile(tempIdFile);
                using (var idPage = ocr.Process(idImg)) {
                    string idText = idPage.GetText();
                    var idMatch = Regex.Match(idText, @"Member\s*ID[:\s]*([A-Za-z0-9\-]+)");
                    if (idMatch.Success)
                    {
                        Console.WriteLine($"Member ID: {idMatch.Groups[1].Value.Trim()}");
                        break; 
                    }
                    else
                    {
                        Console.WriteLine("ID Number not found! Continuing...");
                    }
                }
            }

            if (Cv2.WaitKey(1) == 27) break; // ESC key to exit
        }
    }
}




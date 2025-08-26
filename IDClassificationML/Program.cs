using IDClassificationML; 
using OpenCvSharp;
using System;
using System.IO;

class Program
{
    static void Main(string[] args)
    {
        using var capture = new VideoCapture(0); 
        using var window = new Window("Live Camera");

        if (!capture.IsOpened())
        {
            Console.WriteLine("Camera not detected!");
            return;
        }

        var frame = new Mat();

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
                Console.WriteLine("Valid ID detected with high confidence, stopping...");
                break;
            }

            if (Cv2.WaitKey(1) == 27) break;
        }
    }
}


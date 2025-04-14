### Write a short recap of the four tracking steps and what you implemented there (EKF, track management, data association, camera-lidar sensor fusion). Which results did you achieve? Which part of the project was most difficult for you to complete, and why?
I implemented an EKF for motion prediction, managed tracks with scoring logic, associated measurements using Mahalanobis distance, and added nonlinear camera-lidar sensor fusion. Each step improved the tracking robustness and multi-object capability.

### Do you see any benefits in camera-lidar fusion compared to lidar-only tracking (in theory and in your concrete results)?
Yes, camera-lidar fusion improves accuracy and helps when lidar data is sparse or occluded. I observed better continuity and smoother tracking in fused results.

### Which challenges will a sensor fusion system face in real-life scenarios? Did you see any of these challenges in the project?
Implementing the Jacobian for the camera measurement model was the hardest part. It required careful handling of projection math and potential division by zero.

### Can you think of ways to improve your tracking results in the future?
Using deep learning-based data association or adaptive score management could improve accuracy. Also, incorporating more sensor types may enhance robustness.

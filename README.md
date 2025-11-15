Project flow
<img width="1166" height="296" alt="image" src="https://github.com/user-attachments/assets/feca2de4-6bfa-4af4-8d1c-e8bca65dd836" />

Business case: 
- Company X uses tool "J" to handle tickets from users.
- Usually user send a mail with description of a problem to "J"'s mailbox.
- "J" creates ticket from user's mail automatically
- Y team is involved to sort and dispatch tickets from initial stage to proper service line for further processing

Solution:
- Main purpose of solution is to automate ticket's assignment to responsible service line
- DistilBert NN was fine-tuned to classification (19 classes).
- In: text, out: class of service line prediction

Deploy:
- For test purpose solution was deployed with Flask framework and
- Virtual machine was used to run container with application. Image was upload to VM from DockerHub
- Input for model is orginized via web application

Features:
- Trained model and settings, as well as other functionalities, are packed to image and post to DockerHub
- DockerHub link: docker pull livinmax/distilbert-app:fin
- Code for model training in file train_distilbert.ipynb

Project structure:
- Dockerfile
- app.py
- requirement.txt
- templates
   - index.html

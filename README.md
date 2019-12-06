# QGen_with_gpt2

!python extract_answers.py --key $KEY

!python interact.py --model_checkpoint gpt2_corefs_question_generation --filename temp/$KEY/input.pkl --model_type gpt2 --key $KEY


Run the above 2 commands in the file directory

-> Understanding $KEY and inputs

The input and output are stored in the temp/keyname
In the temp folder make a folder with arbitrary name. In my case I created the folder keyname. So, now $KEY = keyname. Now in the above 
commands replace $KEY by that keyname.
Before using the command for extract answers make sure that you have your input file named metadata.json and replace the paragraph in 
there with your own custom paragraph.
The first command makes a new input.pkl file in the same place.
After the second command you will get the output file as generated_questions.json


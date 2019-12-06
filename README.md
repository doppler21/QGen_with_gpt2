!python extract_answers.py --key $KEY

!python interact.py --model_checkpoint gpt2_corefs_question_generation --filename temp/$KEY/input.pkl --model_type gpt2 --key $KEY

Run the above 2 commands in the file directory

IMPORTANT:
There was an error during upload of folder temp
What to do:
Make a folder named temp
Inside it make a folder named keyname
Inside it store the 3 files metadata.json, input.pkl and questions_generated.json which are present in the temp folder provided.

-> Understanding $KEY and inputs

The input and output are stored in the temp/keyname In the temp folder make a folder with arbitrary name. In my case I created the folder keyname. So, now $KEY = keyname. Now in the above commands replace $KEY by that keyname. Before using the command for extract answers make sure that you have your input file named metadata.json and replace the paragraph in there with your own custom paragraph. The first command makes a new input.pkl file in the same place. After the second command you will get the output file as generated_questions.json

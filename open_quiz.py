
import json

with open('QUIZ_EQUAL_NEW_test.json') as f:
	quiz = json.loads( next(f) )

while 'type' not in quiz:
	print('\n\n')
	for k, choice in enumerate(quiz):
		print('Choice '+str(k)+':'+ ', '.join(word['text'] for word in choice['words']) +'\n')
	choice_index = input('Choice?')
	quiz = quiz[int(choice_index)]['choice']

print('\n\n\n\nQuiz Results - Top Occupations:'+'\n\n'+ '\n'.join(quiz['results']) )
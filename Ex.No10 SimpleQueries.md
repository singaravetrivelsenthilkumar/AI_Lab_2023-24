# Ex.No: 10  Logic Programming –  Simple queries from facts and rules
### DATE: 06.09.2025                                                                         
### REGISTER NUMBER : 212222220048
### AIM: 
To write a prolog program to find the answer of query. 
###  Algorithm:
 Step 1: Start the program <br> 
 Step 2: Convert the sentence into First order Logic  <br> 
 Step 3:  Convert the sentence into Horn clause form  <br> 
 Step 4: Add rules and predicates in a program   <br> 
 Step 5:  Pass the query to program. <br> 
 Step 6: Prolog interpreter shows the output and return answer. <br> 
 Step 8:  Stop the program.
### Program:
### Task 1:
Construct the FOL representation for the following sentences <br> 
1.	John likes all kinds of food.  <br> 
2.	Apples are food.  <br> 
3.	Chicken is a food.  <br> 
4.	Sue eats everything Bill eats. <br> 
5.	 Bill eats peanuts  <br> 
   Convert into clause form and Prove that John like Apple by using Prolog. <br> 
### Program:
```
food(apple).
food(vegetable).
eats(bill,peanuts).
alive(bill).
eats(sue,X):-eats(bill,X).
eats(bill).
```



### Output:
![Screenshot (32)](https://github.com/Vikhram-S/AI_Lab_2023-24/assets/146576573/0804bca9-ce70-459f-822b-3de4a4c090bd)


### Task 2:
Consider the following facts and represent them in predicate form: <br>              
1.	Steve likes easy courses. <br> 
2.	Science courses are hard. <br> 
3. All the courses in Have fun department are easy <br> 
4. BK301 is Have fun department course.<br> 
Convert the facts in predicate form to clauses and then prove by resolution: “Steve likes BK301 course”<br> 

### Program:
```
likes(john,X):-food(X).
eats(sue,X):-eats(bill,X).
eats(bill,peanuts).
food(apple).
food(chicken).
```


### Output:
![Screenshot (33)](https://github.com/Vikhram-S/AI_Lab_2023-24/assets/146576573/318f38c1-c5b8-40b2-b523-f55bd74ba528)




### Task 3:
Consider the statement <br> 
“This is a crime for an American to sell weapons to hostile nations. The Nano , enemy of America has some missiles and its missiles were sold it by Colonal West who is an American” <br> 
Convert to Clause form and prove west is criminal by using Prolog.<br> 
### Program:
```
criminal(X):-
	american(X),
	weapon(Y),
	hostile(Z),
	sells(X,Y,Z).
weapon(Y):-
    missile(Y).
hostile(Z):-
    enemy(Z,X).
sells(west,Y,nano):-
    missile(Y),
    owns(nano,Y).
missile(m).
owns(nano,m).
enemy(nano,america).
american(west).
```
### Output:
![Screenshot (34)](https://github.com/Vikhram-S/AI_Lab_2023-24/assets/146576573/aa255296-7016-48c3-bf05-336df6bcc1d0)


### Result:
Thus the prolog programs were executed successfully and the answer of query was found.

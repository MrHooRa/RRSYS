import csv
from sklearn import tree
import graphviz
from sklearn.datasets import load_iris

X = [[5, 1, 0, 1, 1, 1, 0, 12, 40, 3, 1, 0, 1, 1, 0, 4],
     [4, 1, 0, 1, 0, 1, 0, 23, 30, 3, 0, 0, 0, 0, 0, 4],
     [4, 3, 0, 0, 0, 1, 0, 21, 18, 3, 0, 0, 0, 0, 0, 4],
     [4, 1, 0, 0, 1, 0, 0, 23, 20, 3, 1, 0, 1, 0, 0, 3],
     [3, 3, 1, 0, 0, 0, 0, 31, 24, 2, 1, 1, 1, 1, 0, 3],
     [4, 1, 0, 0, 3, 0, 0, 18, 15, 3, 0, 0, 1, 1, 0, 3],
     [4, 1, 1, 0, 1, 0, 0, 10, 15, 2, 1, 0, 1, 1, 0, 5],
     [3, 1, 0, 0, 1, 0, 0, 26, 28, 2, 0, 0, 0, 0, 1, 3],
     [5, 1, 0, 1, 1, 0, 0, 15, 20, 4, 1, 1, 1, 1, 0, 4],
     [4, 1, 0, 0, 0, 1, 0, 14, 10, 3, 0, 0, 0, 1, 1, 4],
     [4, 1, 0, 0, 0, 1, 0, 24, 20, 3, 1, 0, 1, 1, 1, 3],
     [5, 1, 1, 0, 0, 0, 0, 18, 30, 3, 1, 0, 1, 1, 1, 4],
     [5, 2, 0, 0, 0, 0, 1, 19, 19, 3, 1, 0, 1, 1, 0, 5],
     [4, 3, 0, 0, 0, 1, 1, 25, 23, 3, 1, 0, 1, 1, 1, 3],
     [5, 2, 0, 0, 0, 0, 1, 55, 35, 2, 1, 1, 1, 1, 0, 5],
     [5, 2, 0, 0, 0, 0, 1, 45, 45, 2, 1, 1, 1, 1, 0, 5],
     [4, 2, 0, 0, 0, 0, 1, 40, 30, 2, 1, 1, 1, 1, 0, 5],
     [5, 2, 1, 1, 1, 1, 1, 45, 40, 3, 1, 1, 1, 1, 0, 5],
     [3, 3, 0, 0, 0, 0, 1, 40, 30, 3, 1, 1, 1, 1, 0, 4],
     [4, 2, 1, 0, 1, 1, 1, 35, 30, 4, 1, 1, 1, 1, 0, 4],
     [4, 2, 0, 0, 0, 0, 1, 42, 30, 4, 1, 1, 1, 1, 0, 5],
     [4, 3, 1, 1, 1, 1, 0, 12, 25, 4, 0, 0, 1, 1, 1, 4],
     [4, 3, 1, 1, 1, 1, 0, 14, 25, 4, 0, 0, 1, 1, 1, 4],
     [4, 3, 0, 0, 0, 1, 0, 24, 20, 2, 1, 0, 1, 1, 1, 4],
     [4, 2, 0, 0, 0, 1, 1, 70, 30, 2, 1, 1, 1, 1, 0, 5],
     [4, 3, 0, 0, 0, 1, 0, 40, 45, 3, 0, 0, 1, 1, 1, 5],
     [3, 3, 0, 0, 0, 1, 0, 40, 30, 3, 0, 0, 1, 1, 0, 4],
     [4, 3, 0, 0, 0, 0, 1, 30, 25, 1, 1, 0, 1, 1, 0, 4],
     [3, 1, 0, 0, 0, 1, 0, 16, 30, 3, 1, 0, 1, 1, 1, 2],
     [3, 1, 1, 1, 1, 1, 0, 14, 18, 4, 0, 0, 1, 1, 1, 2],
     [4, 1, 0, 0, 1, 0, 0, 14, 15, 3, 0, 0, 1, 1, 1, 4],
     [4, 1, 1, 0, 0, 0, 0, 30, 15, 2, 0, 0, 0, 0, 1, 4],
     [4, 3, 0, 0, 0, 1, 1, 22, 18, 3, 1, 0, 1, 1, 1, 3],
     [3, 1, 0, 0, 1, 0, 0, 18, 14, 3, 0, 0, 0, 0, 1, 4],
     [4, 1, 0, 0, 0, 1, 0, 5, 15, 3, 0, 0, 1, 1, 1, 3],
     [4, 1, 0, 0, 0, 0, 1, 12, 45, 3, 1, 0, 0, 1, 0, 5],
     [5, 1, 0, 0, 0, 0, 1, 25, 35, 3, 1, 1, 1, 1, 0, 4]
     ]
Y = [['albaik'],
     ['KFC'],
     ['Mcdonalds'],
     ['Shawrmar'],
     ['Pizza_hot'],
     ['Shawrma_clasic'],
     ['Manousha_mega'],
     ['Shawrma_plus'],
     ['Algandol'],
     ['Bofeah_ibrahem'],
     ['Burger_king'],
     ['Fatira_falafl'],
     ['Sub-way'],
     ['Kudo'],
     ['Armin'],
     ['Almakolat_albahrya'],
     ['Big_chifs'],
     ['Blue_gardean'],
     ['Kober_shendny'],
     ['Vairos_Gardean'],
     ['The_gril'],
     ['Mama_noura'],
     ['Bet_alshawrma'],
     ['High_way_5'],
     ['Sutlan_staek_house'],
     ['Claivornea_burger'],
     ['Five_guys'],
     ['Ihop'],
     ['Herfy'],
     ['Imratot_alshawrma'],
     ['Fatira_alshawrma'],
     ['Maestro_pizza'],
     ['The_sandwitch_compny'],
     ['Nakhat_alshawar'],
     ['Burger_aldek'],
     ['Zeton_and_tean'],
     ['Mamz_bery_baery']]

userPr = []
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
print("\n")

for i in range(len(Y)):
    print(f"\t({i+1}): {Y[i][0]}")



# add user re want into a list
# for i in range(len(askUserToChoose)):
#     if (i+1) <= len(askUserToChoose):
#         if askUserToChoose[(i+1)] == ",":    
#             if askUserToChoose[i] == ",":
#                 continue
t = True
print("\n")
while(t):
    askUserToChoose = input("Choose resturant number (-1 to stop): ")
    if int(askUserToChoose) == -1:
        t = False
        break
    userPr.append((int(askUserToChoose) - 1))


# Print user re name
print("Your best restaurant is/are:", end =" ")
for i in range(len(userPr)):
    print("(", Y[userPr[i]][0], ")" , end =" ")
print("\n")
# replace re number to re data
for i in range(len(userPr)):
    userPr[i] = X[userPr[i]]


# calculation restaurants data to make average data
cal = []
temp = 0
for i in range(16):
    for j in range(len(userPr)):
        temp += userPr[j][i]
    cal.append(temp)
    # print(f"Total variable ({i+1}): = {temp}")
    temp = 0

print("\nAll restaurants variables", cal)
n = len(userPr)
for i in range(len(cal)):
    cal[i] = round((cal[i] / n))
print("After calculation all restaurants data", cal)


predict = clf.predict([cal])
print(f"\n\tBest predict is: {predict[0]}")

# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)

# dot_data = tree.export_graphviz(clf, out_file=None,
#                                 filled=True, rounded=True,
#                                 special_characters=True)
# graph = graphviz.Source(dot_data)
# graph

input()

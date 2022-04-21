Skip to left side bar




Filter files by name
/Desktop/Citizens_Income_Prediction/
Name
Last Modified
148
ac_dt = accuracy_score(y_test,dt)
149
ac_rf = accuracy_score(y_test,rdf)
150
​
151
​
152
# In[12]:
153
​
154
​
155
scores = [ac_log, ac_knn, ac_nb, ac_svm, ac_dt, ac_rf]
156
​
157
​
158
# In[13]:
159
​
160
​
161
algorithms = ['Logistic Regression' ,
162
              'K-Nearest Neighbors',
163
              'Naive Bayes',
164
              'Support Vector Machine',
165
             'Decision Tree',
166
             'Random Forest Classifier']
167
​
168
​
169
# In[14]:
170
​
171
​
172
sc_max_y_lim = max(scores) + 0.05
173
sc_min_y_lim = min(scores) - 0.05
174
​
175
pr_max_y_lim = max(precisions) + 0.05
176
pr_min_y_lim = min(precisions) - 0.05
177
​
178
re_max_y_lim = max(recalls) + 0.05
179
re_min_y_lim = min(recalls) - 0.05
180
​
181
f1_max_y_lim = max(f1_scores) + 0.05
182
f1_min_y_lim = min(f1_scores) - 0.05
183
​
184
​
185
# In[15]:
186
​
187
​
188
fig=plt.figure(figsize=(10,25))
189
​
190
plt.subplot(4,1,1)
191
plt.xlim(sc_min_y_lim, sc_max_y_lim)
192
bars =plt.barh(algorithms, scores)
193
plt.bar_label(bars)
194
plt.xlabel("Algorithms")
195
plt.ylabel('Accuracy score')
196
plt.title('Accuracy Score Bar Plot')
197
​
198
plt.subplot(4,1,2)
199
plt.xlim(pr_min_y_lim, pr_max_y_lim)
200
bars =plt.barh(algorithms, precisions)
201
plt.bar_label(bars)
202
plt.xlabel("Algorithms")
203
plt.ylabel('Precision')
204
plt.title('Precision Bar Plot')
205
​
206
plt.subplot(4,1,3)
207
plt.xlim(re_min_y_lim, re_max_y_lim)
208
bars =plt.barh(algorithms, recalls)
209
plt.bar_label(bars)
210
plt.xlabel("Algorithms")
211
plt.ylabel('Recall')
212
plt.title('Recall Bar Plot')
213
​
214
plt.subplot(4,1,4)
215
plt.xlim(f1_min_y_lim, f1_max_y_lim)
216
bars =plt.barh(algorithms, f1_scores)
217
plt.bar_label(bars)
218
plt.xlabel("Algorithms")
219
plt.ylabel('F1 score')
220
plt.title('F1 Score Bar Plot')
221
​
222
plt.show()
223
​
224
​

Simple
3
30
Python
Model_Comparision.py
Spaces: 4
Ln 1, Col 1

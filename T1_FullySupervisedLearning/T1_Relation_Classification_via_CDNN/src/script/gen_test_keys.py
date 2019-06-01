in_file = ('data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/' + 
         'TEST_FILE_FULL.TXT')
out_file = 'data/test_keys.txt'

with open(in_file) as in_f:
  with open(out_file, 'w') as out_f:
    lines  = in_f.readlines()
    n = int(len(lines)/4)
    print(len(lines), n)

    for i in range(n):
      id_str = lines[4*i].split()[0]
      label = lines[4*i+1]
      out_f.write(id_str + '\t' +label)
      

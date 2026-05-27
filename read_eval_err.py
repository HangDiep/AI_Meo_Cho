encs=['utf-8','utf-16','utf-16-le','utf-16-be','latin-1']
for enc in encs:
    try:
        with open('eval_err.txt','rb') as f:
            b=f.read()
        s=b.decode(enc)
    except Exception as e:
        print('ENC',enc,'DECODE_ERROR',e)
        continue
    print('\n===== encoding:',enc,'=====\n')
    print(s[-4000:])

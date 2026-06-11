# Traffic Annotation Guide

Bo nhan dung 5 class ID co dinh:

```text
0 person
1 car
2 motorcycle
3 bus
4 truck
```

## Quy tac chung

- Ve box sat vat the, khong gom bong do.
- Nhan tat ca vat the ro rang trong ROI, khong chi vat the cat qua vach.
- Vat the bi che mot phan van duoc nhan neu con nhin thay du hinh dang.
- Bo qua vat the qua nho, mo hoac chi thay mot phan rat nho.
- Khong gan nhan dua tren suy doan neu khong phan biet duoc class.
- Khong thay doi thu tu class trong qua trinh gan nhan.

## Person

- Chi dung cho nguoi.
- Nguoi lai xe may van co mot box `person`.
- Neu nguoi ngoi trong o to/bus va khong thay ro toan than thi khong can nhan.

## Motorcycle

- Ve box quanh xe may, khong mo rong box de bao het nguoi lai.
- Nguoi lai xe may duoc nhan rieng bang box `person`.
- Xe tay ga, mo to va xe may pho thong deu la `motorcycle`.
- Khong gan tuk-tuk/xe ba banh vao `motorcycle`; bo qua trong bo pilot nay.

## Car

- Sedan, hatchback, SUV, taxi va van nho la `car`.
- Box quanh than xe, khong gom bong do.

## Bus

- Xe buyt thanh pho, coach va minibus lon la `bus`.
- Minivan nho van la `car`.

## Truck

- Xe tai, pickup cho hang ro rang va xe chuyen dung cho hang la `truck`.
- Pickup dung nhu xe con co the gan `car`; can giu quy tac nhat quan.

## Occlusion va truncation

- Neu vat the con thay khoang 20% tro len va class van ro, co the gan box quanh
  phan thay duoc.
- Neu vat the bi cat boi bien anh, ve box den sat bien anh.
- Neu hai vat the chong len nhau, moi vat the co box rieng.

## Kiem tra chat luong

Truoc khi export, kiem tra:

- Khong co box rong hoac ra ngoai anh.
- Class ID nam trong khoang 0-4.
- Khong bo sot xe may gan camera.
- Rider va motorcycle la hai box rieng.
- Cac frame lien tiep dung cung mot quy tac class.

Export theo YOLO format:

```text
class_id center_x center_y width height
```

Bon toa do phai duoc chuan hoa ve khoang 0-1.

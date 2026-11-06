# Hasil Pengujian /ask (http://localhost:8000)

## 1. FAQ - kebijakan cuti — PASS
- Query: `Berapa jatah cuti tahunan karyawan tetap?`
- Diharapkan: intent=`faq`, status=`success`
- HTTP 200
```json
{
  "status": "success",
  "intent": "faq",
  "data": "Karyawan tetap mendapat jatah 12 hari cuti tahunan yang dapat diambil setelah masa kerja minimal 3 bulan."
}
```

## 2. FAQ - reset password VPN — PASS
- Query: `Bagaimana cara reset password VPN perusahaan?`
- Diharapkan: intent=`faq`, status=`success`
- HTTP 200
```json
{
  "status": "success",
  "intent": "faq",
  "data": "Reset password VPN dapat dilakukan secara mandiri melalui portal self-service IT. Apabila akun Anda terkunci, silakan hubungi tim IT Support untuk bantuan lebih lanjut."
}
```

## 3. Tiket ada — PASS
- Query: `Apa status tiket TICKET-0005 saya?`
- Diharapkan: intent=`ticket_status`, status=`success`
- HTTP 200
```json
{
  "status": "success",
  "intent": "ticket_status",
  "data": "Tiket TICKET-0005 — \"Reset password VPN gagal terus\" (kategori: Access). Status: Diproses. Update terakhir: 2026-08-27."
}
```

## 4. Tiket tidak ada — PASS
- Query: `Tolong cek status tiket TICKET-0042`
- Diharapkan: intent=`ticket_status`, status=`not_found`
- HTTP 200
```json
{
  "status": "not_found",
  "intent": "ticket_status",
  "data": "Tiket TICKET-0042 tidak ditemukan. Periksa kembali ID tiket Anda (format TICKET-XXXX) atau hubungi tim IT Support."
}
```

## 5. Di luar cakupan — PASS
- Query: `Bagaimana prediksi cuaca di Jakarta besok?`
- Diharapkan: intent=`faq`, status=`success`
- HTTP 200
```json
{
  "status": "success",
  "intent": "faq",
  "data": "Maaf, saya tidak menemukan informasi tersebut di knowledge base. Silakan hubungi tim IT Support atau HR untuk bantuan lebih lanjut."
}
```

## 6. Tambahan - minta cek tiket tanpa ID — PASS
- Query: `Tolong cek status tiket saya`
- Diharapkan: intent=`ticket_status`, status=`error`
- HTTP 200
```json
{
  "status": "error",
  "intent": "ticket_status",
  "data": "Saya mendeteksi Anda ingin mengecek status tiket, tetapi ID tiket tidak ditemukan. Sertakan ID dengan format TICKET-XXXX, contoh: TICKET-0005."
}
```

## 7. Tambahan - input kosong spasi — PASS
- Query: `   `
- Diharapkan: intent=`unknown`, status=`error`
- HTTP 200
```json
{
  "status": "error",
  "intent": "unknown",
  "data": "Maaf, saya tidak memahami permintaan Anda. Silakan ajukan pertanyaan seputar kebijakan/IT, atau sertakan ID tiket (TICKET-XXXX) untuk cek status."
}
```

**7/7 skenario sesuai ekspektasi routing.**

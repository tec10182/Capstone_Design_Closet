import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:closet/services/upload_service.dart';

class UploadScreen extends StatefulWidget {
  final String userId;

  const UploadScreen({required this.userId, super.key});

  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  bool isLoading = false; // 로딩 상태 플래그

  Future<void> pickAndUploadImage(BuildContext context) async {
    final ImagePicker picker = ImagePicker();
    final XFile? pickedFile =
        await picker.pickImage(source: ImageSource.gallery);

    if (pickedFile == null) {
      // 사용자가 이미지를 선택하지 않고 취소한 경우 처리
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("No image selected.")),
      );
      return;
    }

    final File image = File(pickedFile.path);
    setState(() {
      isLoading = true; // 로딩 상태 시작
    });

    try {
      // 업로드 서비스 호출 및 AI 분류 결과 가져오기
      final String? category = await UploadService.uploadImageAndClassify(
        image,
        widget.userId,
      );

      if (category != null && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Image uploaded successfully!")),
        );
        Navigator.pop(context, {"image": pickedFile.path, "type": category});
      } else if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text("Image upload failed.")),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Error: ${e.toString()}")),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          isLoading = false; // 로딩 상태 종료
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Upload Item")),
      body: Center(
        child: isLoading
            ? const CircularProgressIndicator() // 로딩 중일 때 표시
            : ElevatedButton(
                onPressed: () => pickAndUploadImage(context),
                child: const Text("Pick and Upload"),
              ),
      ),
    );
  }
}

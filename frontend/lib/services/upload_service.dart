import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:logger/logger.dart';

class UploadService {
  static final Logger _logger = Logger();

  static Future<String?> uploadImageAndClassify(
      File image, String userId) async {
    final url = Uri.parse('http://127.0.0.1:8000/api/v1/upload/img');
    final request = http.MultipartRequest('POST', url)
      ..fields['id'] = userId
      ..files.add(await http.MultipartFile.fromPath('file', image.path));

    try {
      final response = await request.send();

      if (response.statusCode == 200) {
        final responseBody = await response.stream.bytesToString();
        final Map<String, dynamic> data = jsonDecode(responseBody);

        if (data['success'] == true && data.containsKey('category')) {
          return data['category'] as String; // 서버에서 반환된 카테고리 값
        }
      } else {
        _logger
            .e("Failed to upload image. Status code: ${response.statusCode}");
      }
    } catch (e) {
      _logger.e("Error during upload", e);
    }

    return null; // 실패 시 null 반환
  }
}

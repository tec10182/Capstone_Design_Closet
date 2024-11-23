import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class RecommendService {
  static const String baseUrl = 'http://127.0.0.1:8000/api/v1';

  // 사용자 옷장 데이터 가져오기
  static Future<Map<String, dynamic>> fetchCloset(String userId) async {
    final url = Uri.parse('$baseUrl/user/image');
    final response = await http.post(
      url,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'id': userId}),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to fetch closet data');
    }
  }

  // 추천 점수 계산
  static Future<Map<String, dynamic>> calculateScore(
      File image, String userId) async {
    final url = Uri.parse('$baseUrl/compatibility/score');
    final request = http.MultipartRequest('POST', url)
      ..fields['id'] = userId
      ..files.add(await http.MultipartFile.fromPath('file', image.path));

    final response = await request.send();
    if (response.statusCode == 200) {
      final responseBody = await response.stream.bytesToString();
      return jsonDecode(responseBody);
    } else {
      throw Exception('Failed to calculate score');
    }
  }
}

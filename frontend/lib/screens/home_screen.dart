import 'package:flutter/material.dart';
import 'package:closet/services/recommend_service.dart';
import 'package:closet/screens/upload_screen.dart';

class HomeScreen extends StatefulWidget {
  final String userId;

  const HomeScreen({required this.userId, super.key});

  @override
  HomeScreenState createState() => HomeScreenState();
}

class HomeScreenState extends State<HomeScreen> {
  List<String> tops = [];
  List<String> bottoms = [];
  bool isLoading = true;

  @override
  void initState() {
    super.initState();
    fetchClosetData();
  }

  Future<void> fetchClosetData() async {
    try {
      final closetData = await RecommendService.fetchCloset(widget.userId);
      if (mounted) {
        setState(() {
          tops = List<String>.from(closetData['tops']);
          bottoms = List<String>.from(closetData['bottoms']);
          isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Error fetching closet: ${e.toString()}")),
        );
      }
    }
  }

  // 추가 항목 처리 함수
  void addItem(String imagePath, String type) {
    setState(() {
      if (type == "Top") {
        tops.add(imagePath); // Top 리스트에 추가
      } else if (type == "Bottom") {
        bottoms.add(imagePath); // Bottom 리스트에 추가
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          "CLOSET",
          style: TextStyle(
            color: Colors.black,
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
        centerTitle: true,
        backgroundColor: Colors.white,
        elevation: 0,
        actions: [
          IconButton(
            onPressed: () {
              // 검색 기능
            },
            icon: const Icon(Icons.search, color: Colors.black),
          ),
          IconButton(
            onPressed: () {
              // 체크 기능
            },
            icon: const Icon(Icons.check, color: Colors.black),
          ),
        ],
      ),
      body: isLoading
          ? const Center(child: CircularProgressIndicator())
          : tops.isEmpty && bottoms.isEmpty
              ? const Center(
                  child: Text(
                    "Upload your clothes",
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                )
              : SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      buildCategory("Tops", tops),
                      const SizedBox(height: 20),
                      buildCategory("Bottoms", bottoms),
                    ],
                  ),
                ),
      bottomNavigationBar: BottomNavigationBar(
        type: BottomNavigationBarType.fixed,
        backgroundColor: Colors.black,
        selectedItemColor: Colors.white,
        unselectedItemColor: Colors.grey,
        currentIndex: 0,
        onTap: (index) {
          if (index == 1) {
            // Upload 버튼 클릭 시 UploadScreen으로 이동
            Navigator.push(
              context,
              MaterialPageRoute(
                builder: (context) => UploadScreen(userId: widget.userId),
              ),
            ).then((result) {
              if (result != null) {
                final Map<String, String> uploadedItem =
                    result as Map<String, String>;
                addItem(uploadedItem["image"]!, uploadedItem["type"]!);
              }
            });
          }
        },
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.home),
            label: 'Home',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.add), // 업로드 화면
            label: 'Upload',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.thumb_up), // 추천 화면
            label: 'Recommend',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.settings), // 설정 화면
            label: 'Settings',
          ),
        ],
      ),
    );
  }

  Widget buildCategory(String title, List<String> items) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 10),
        GridView.builder(
          physics: const NeverScrollableScrollPhysics(),
          shrinkWrap: true,
          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 2,
            crossAxisSpacing: 10,
            mainAxisSpacing: 10,
            childAspectRatio: 3 / 4,
          ),
          itemCount: items.length,
          itemBuilder: (context, index) {
            return ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Image.network(
                items[index],
                fit: BoxFit.cover,
              ),
            );
          },
        ),
      ],
    );
  }
}
